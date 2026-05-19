use std::env;
use std::fs::File;
use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use hdf5_reader::{H5Type, Hdf5File, SliceInfo, SliceInfoElem};
use rig_vector_testkit::{
    AnnFixture, AnnFixtureSource, AnnMetric, FixtureDocument, FixtureQuery, SourceGroundTruth,
    SourceNeighbor, compute_expected_neighbors, score_from_distance,
};

fn main() -> Result<()> {
    let args = Args::parse()?;
    let fixture = build_fixture(&args)?;

    let output = File::create(&args.output)
        .with_context(|| format!("failed to create '{}'", args.output.display()))?;
    serde_json::to_writer_pretty(output, &fixture).context("failed to write fixture JSON")?;

    Ok(())
}

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    name: String,
    dataset: String,
    source_kind: String,
    url: Option<String>,
    source_metric: Option<String>,
    metric: AnnMetric,
    documents: usize,
    queries: usize,
    top_k: usize,
    threshold_keep: Option<usize>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut input = None;
        let mut output = None;
        let mut name = None;
        let mut dataset = None;
        let mut source_kind = "ann-benchmarks-hdf5".to_string();
        let mut url = None;
        let mut source_metric = None;
        let mut metric = None;
        let mut documents = 16usize;
        let mut queries = 3usize;
        let mut top_k = 5usize;
        let mut threshold_keep = Some(3usize);

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => input = Some(PathBuf::from(next_value(&mut args, "--input")?)),
                "--output" => output = Some(PathBuf::from(next_value(&mut args, "--output")?)),
                "--name" => name = Some(next_value(&mut args, "--name")?),
                "--dataset" => dataset = Some(next_value(&mut args, "--dataset")?),
                "--source-kind" => source_kind = next_value(&mut args, "--source-kind")?,
                "--url" => url = Some(next_value(&mut args, "--url")?),
                "--source-metric" => {
                    source_metric = Some(next_value(&mut args, "--source-metric")?)
                }
                "--metric" => metric = Some(parse_metric(&next_value(&mut args, "--metric")?)?),
                "--documents" => {
                    documents = parse_usize(&next_value(&mut args, "--documents")?, "--documents")?
                }
                "--queries" => {
                    queries = parse_usize(&next_value(&mut args, "--queries")?, "--queries")?
                }
                "--top-k" => top_k = parse_usize(&next_value(&mut args, "--top-k")?, "--top-k")?,
                "--threshold-keep" => {
                    threshold_keep = Some(parse_usize(
                        &next_value(&mut args, "--threshold-keep")?,
                        "--threshold-keep",
                    )?)
                }
                "--no-threshold" => threshold_keep = None,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => bail!("unknown argument '{other}'"),
            }
        }

        let input = input.context("--input is required")?;
        let output = output.context("--output is required")?;
        let name = name.context("--name is required")?;
        let dataset = dataset.context("--dataset is required")?;
        let metric = metric.context("--metric is required")?;

        ensure!(
            !source_kind.trim().is_empty(),
            "--source-kind must not be empty"
        );
        ensure!(documents > 0, "--documents must be greater than zero");
        ensure!(queries > 0, "--queries must be greater than zero");
        ensure!(top_k > 0, "--top-k must be greater than zero");
        ensure!(
            top_k <= documents,
            "--top-k must be less than or equal to --documents"
        );
        if let Some(threshold_keep) = threshold_keep {
            ensure!(
                threshold_keep > 0 && threshold_keep < top_k,
                "--threshold-keep must be greater than zero and less than --top-k"
            );
        }

        Ok(Self {
            input,
            output,
            name,
            dataset,
            source_kind,
            url,
            source_metric,
            metric,
            documents,
            queries,
            top_k,
            threshold_keep,
        })
    }
}

fn build_fixture(args: &Args) -> Result<AnnFixture> {
    let file = Hdf5File::open(&args.input)
        .with_context(|| format!("failed to open '{}'", args.input.display()))?;
    let train = file.dataset("/train").context("missing /train dataset")?;
    let test = file.dataset("/test").context("missing /test dataset")?;
    let (train_total, train_dimensions) = dense_dataset_shape(&train, "/train")?;
    let (test_total, test_dimensions) = dense_dataset_shape(&test, "/test")?;
    ensure!(
        train_dimensions == test_dimensions,
        "train dimensions {train_dimensions} did not match test dimensions {test_dimensions}"
    );

    let query_vectors =
        read_dense_rows(&test, 0, args.queries).context("failed to read test rows")?;
    let mut source_ground_truth = read_source_ground_truth(
        &file,
        args,
        train_total,
        test_total,
        &std::collections::HashMap::new(),
    )?;
    let train_indices = selected_train_indices(args.documents, source_ground_truth.as_deref());
    let train_vectors = read_dense_rows_for_indices(&train, &train_indices)
        .context("failed to read selected train rows")?;
    let dimensions = train_vectors
        .first()
        .map(Vec::len)
        .context("benchmark train dataset yielded no rows")?;

    ensure!(
        query_vectors.iter().all(|query| query.len() == dimensions),
        "query vectors do not match train vector dimensions"
    );
    let top_k = u64::try_from(args.top_k).context("--top-k did not fit in u64")?;

    let source_prefix = args.source_kind.trim_end_matches("-hdf5");
    let documents = train_vectors
        .iter()
        .zip(train_indices.iter())
        .map(|(vector, source_index)| FixtureDocument {
            id: format!("{source_prefix}:{}:train:{source_index}", args.dataset),
            text: format!("{source_prefix}:{}:train:{source_index}", args.dataset),
            vector: vector.clone(),
        })
        .collect::<Vec<_>>();
    let document_ids_by_source_index = documents
        .iter()
        .zip(train_indices.iter())
        .map(|(document, source_index)| (*source_index, document.id.clone()))
        .collect::<std::collections::HashMap<_, _>>();
    add_source_ground_truth_ids(
        source_ground_truth.as_deref_mut(),
        &document_ids_by_source_index,
    );

    let last_query_index = args.queries.saturating_sub(1);
    let queries = query_vectors
        .iter()
        .enumerate()
        .map(|(query_index, vector)| -> Result<FixtureQuery> {
            let unthresholded =
                compute_expected_neighbors(args.metric, &documents, vector, top_k, None)?;
            let threshold = if query_index == last_query_index {
                threshold_for(&unthresholded, args.threshold_keep)
            } else {
                None
            };
            let expected =
                compute_expected_neighbors(args.metric, &documents, vector, top_k, threshold)?;

            Ok(FixtureQuery {
                id: format!("{source_prefix}:{}:test:{query_index}", args.dataset),
                text: format!("{source_prefix}:{}:test:{query_index}", args.dataset),
                vector: vector.clone(),
                top_k,
                threshold,
                expected,
                source_ground_truth: source_ground_truth
                    .as_ref()
                    .map(|ground_truth| {
                        ground_truth.get(query_index).cloned().with_context(|| {
                            format!("missing source ground truth row for query {query_index}")
                        })
                    })
                    .transpose()?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    AnnFixture::new(
        args.name.clone(),
        args.metric,
        dimensions,
        Some(AnnFixtureSource {
            kind: args.source_kind.clone(),
            dataset: args.dataset.clone(),
            url: args.url.clone(),
            source_metric: args.source_metric.clone(),
            train_start: 0,
            train_count: train_indices.len(),
            train_indices: Some(train_indices),
            train_total: Some(train_total),
            test_start: 0,
            test_count: args.queries,
            test_total: Some(test_total),
            top_k: args.top_k,
            generated_by: "cargo run -p rig-vector-testkit --features ann-benchmarks --bin hdf5_vector_fixture".to_string(),
        }),
        documents,
        queries,
    )
}

fn read_dense_rows(
    dataset: &hdf5_reader::Dataset,
    start: u64,
    row_count: usize,
) -> Result<Vec<Vec<f64>>> {
    let (total_rows, dimensions) = dense_dataset_shape_u64(dataset, "dense dataset")?;
    let row_count_u64 = u64::try_from(row_count).context("row count did not fit in u64")?;
    let end = start
        .checked_add(row_count_u64)
        .context("row selection overflowed")?;
    ensure!(
        end <= total_rows,
        "requested rows {start}..{end}, but dataset only has {total_rows} rows"
    );

    let selection = SliceInfo {
        selections: vec![
            SliceInfoElem::Slice {
                start,
                end,
                step: 1,
            },
            SliceInfoElem::Slice {
                start: 0,
                end: dimensions,
                step: 1,
            },
        ],
    };
    read_dense_rows_with_type::<f32>(dataset, &selection, row_count, dimensions)
        .or_else(|f32_error| {
            read_dense_rows_with_type::<f64>(dataset, &selection, row_count, dimensions)
                .with_context(|| format!("failed to read dense rows as f32: {f32_error}"))
        })
        .or_else(|float_error| {
            read_dense_rows_with_type::<u8>(dataset, &selection, row_count, dimensions)
                .with_context(|| format!("failed to read dense rows as float data: {float_error}"))
        })
}

fn read_dense_rows_for_indices(
    dataset: &hdf5_reader::Dataset,
    indices: &[usize],
) -> Result<Vec<Vec<f64>>> {
    indices
        .iter()
        .map(|index| {
            let start = u64::try_from(*index).context("source train index did not fit in u64")?;
            let mut rows = read_dense_rows(dataset, start, 1)
                .with_context(|| format!("failed to read train row {index}"))?;
            rows.pop()
                .with_context(|| format!("train row {index} yielded no vector"))
        })
        .collect()
}

fn selected_train_indices(
    prefix_count: usize,
    source_ground_truth: Option<&[SourceGroundTruth]>,
) -> Vec<usize> {
    let mut indices = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for index in 0..prefix_count {
        if seen.insert(index) {
            indices.push(index);
        }
    }
    if let Some(source_ground_truth) = source_ground_truth {
        for source_index in source_ground_truth
            .iter()
            .flat_map(|ground_truth| ground_truth.neighbors.iter())
            .map(|neighbor| neighbor.source_index)
        {
            if seen.insert(source_index) {
                indices.push(source_index);
            }
        }
    }

    indices
}

fn add_source_ground_truth_ids(
    source_ground_truth: Option<&mut [SourceGroundTruth]>,
    document_ids_by_source_index: &std::collections::HashMap<usize, String>,
) {
    if let Some(source_ground_truth) = source_ground_truth {
        for ground_truth in source_ground_truth {
            for neighbor in &mut ground_truth.neighbors {
                neighbor.id = document_ids_by_source_index
                    .get(&neighbor.source_index)
                    .cloned();
            }
        }
    }
}

fn dense_dataset_shape(dataset: &hdf5_reader::Dataset, name: &str) -> Result<(usize, usize)> {
    let (rows, columns) = dense_dataset_shape_u64(dataset, name)?;
    Ok((
        usize::try_from(rows).context("dataset row count did not fit in usize")?,
        usize::try_from(columns).context("dataset dimension count did not fit in usize")?,
    ))
}

fn dense_dataset_shape_u64(dataset: &hdf5_reader::Dataset, name: &str) -> Result<(u64, u64)> {
    let shape = dataset.shape();
    ensure!(
        shape.len() == 2,
        "expected {name} to be a dense 2-D dataset, got shape {shape:?}"
    );
    let total_rows = shape
        .first()
        .copied()
        .context("dense dataset shape did not include row count")?;
    let dimensions = shape
        .get(1)
        .copied()
        .context("dense dataset shape did not include dimensions")?;
    Ok((total_rows, dimensions))
}

fn read_source_ground_truth(
    file: &Hdf5File,
    args: &Args,
    train_total: usize,
    test_total: usize,
    document_ids_by_source_index: &std::collections::HashMap<usize, String>,
) -> Result<Option<Vec<SourceGroundTruth>>> {
    let (neighbors, distances) = match (
        file.dataset("/neighbors").ok(),
        file.dataset("/distances").ok(),
    ) {
        (Some(neighbors), Some(distances)) => (neighbors, distances),
        (None, None) => return Ok(None),
        _ => bail!("source HDF5 must contain both /neighbors and /distances or neither"),
    };
    let metric = source_metric_to_ann_metric(args.source_metric.as_deref(), args.metric)?;
    let neighbor_shape = shape_2d(&neighbors, "/neighbors")?;
    let distance_shape = shape_2d(&distances, "/distances")?;
    ensure_ground_truth_shapes(
        neighbor_shape,
        distance_shape,
        test_total,
        args.queries,
        args.top_k,
    )?;

    let source_indices = read_index_rows(&neighbors, 0, args.queries, args.top_k)
        .context("failed to read /neighbors")?;
    let source_distances = read_distance_rows(&distances, 0, args.queries, args.top_k)
        .context("failed to read /distances")?;

    let ground_truth = source_indices
        .into_iter()
        .zip(source_distances)
        .enumerate()
        .map(|(query_index, (indices, distances))| -> Result<SourceGroundTruth> {
            let neighbors = indices
                .into_iter()
                .zip(distances)
                .map(|(source_index, distance)| -> Result<SourceNeighbor> {
                    ensure!(
                        source_index < train_total,
                        "source neighbor index {source_index} for query {query_index} exceeds train total {train_total}"
                    );
                    Ok(SourceNeighbor {
                        source_index,
                        id: document_ids_by_source_index.get(&source_index).cloned(),
                        distance,
                        score: score_from_distance(metric, distance)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(SourceGroundTruth {
                metric,
                source_metric: args.source_metric.clone(),
                neighbors,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Some(ground_truth))
}

fn shape_2d(dataset: &hdf5_reader::Dataset, name: &str) -> Result<(usize, usize)> {
    dense_dataset_shape(dataset, name)
}

fn ensure_ground_truth_shapes(
    neighbor_shape: (usize, usize),
    distance_shape: (usize, usize),
    test_total: usize,
    requested_queries: usize,
    requested_top_k: usize,
) -> Result<()> {
    ensure!(
        neighbor_shape == distance_shape,
        "/neighbors shape {neighbor_shape:?} did not match /distances shape {distance_shape:?}"
    );
    ensure!(
        neighbor_shape.0 == test_total,
        "/neighbors row count {} did not match /test row count {test_total}",
        neighbor_shape.0
    );
    ensure!(
        requested_queries <= neighbor_shape.0,
        "requested {requested_queries} source ground-truth queries, but /neighbors has {} rows",
        neighbor_shape.0
    );
    ensure!(
        requested_top_k <= neighbor_shape.1,
        "requested source top_k {requested_top_k}, but /neighbors has only {} columns",
        neighbor_shape.1
    );
    Ok(())
}

fn read_index_rows(
    dataset: &hdf5_reader::Dataset,
    start: u64,
    row_count: usize,
    column_count: usize,
) -> Result<Vec<Vec<usize>>> {
    read_index_rows_with_type::<u64>(dataset, start, row_count, column_count)
        .or_else(|u64_error| {
            read_index_rows_with_type::<u32>(dataset, start, row_count, column_count)
                .with_context(|| format!("failed to read index rows as u64: {u64_error}"))
        })
        .or_else(|unsigned_error| {
            read_index_rows_with_type::<i64>(dataset, start, row_count, column_count).with_context(
                || format!("failed to read index rows as unsigned data: {unsigned_error}"),
            )
        })
        .or_else(|i64_error| {
            read_index_rows_with_type::<i32>(dataset, start, row_count, column_count)
                .with_context(|| format!("failed to read index rows as i64: {i64_error}"))
        })
}

fn read_index_rows_with_type<T>(
    dataset: &hdf5_reader::Dataset,
    start: u64,
    row_count: usize,
    column_count: usize,
) -> Result<Vec<Vec<usize>>>
where
    T: H5Type + Copy + TryInto<usize>,
    <T as TryInto<usize>>::Error: std::fmt::Debug,
{
    let rows = read_matrix_selection_with_type::<T>(dataset, start, row_count, column_count)?;
    rows.into_iter()
        .enumerate()
        .map(|(row_index, row)| {
            row.into_iter()
                .enumerate()
                .map(|(column_index, value)| {
                    value.try_into().map_err(|error| {
                        anyhow::anyhow!(
                            "source neighbor index at row {row_index} column {column_index} was invalid: {error:?}"
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect()
}

fn read_distance_rows(
    dataset: &hdf5_reader::Dataset,
    start: u64,
    row_count: usize,
    column_count: usize,
) -> Result<Vec<Vec<f64>>> {
    read_distance_rows_with_type::<f32>(dataset, start, row_count, column_count).or_else(
        |f32_error| {
            read_distance_rows_with_type::<f64>(dataset, start, row_count, column_count)
                .with_context(|| format!("failed to read distance rows as f32: {f32_error}"))
        },
    )
}

fn read_distance_rows_with_type<T>(
    dataset: &hdf5_reader::Dataset,
    start: u64,
    row_count: usize,
    column_count: usize,
) -> Result<Vec<Vec<f64>>>
where
    T: H5Type + Copy + Into<f64>,
{
    let rows = read_matrix_selection_with_type::<T>(dataset, start, row_count, column_count)?;
    Ok(rows
        .into_iter()
        .map(|row| row.into_iter().map(Into::into).collect())
        .collect())
}

fn read_matrix_selection_with_type<T>(
    dataset: &hdf5_reader::Dataset,
    start: u64,
    row_count: usize,
    column_count: usize,
) -> Result<Vec<Vec<T>>>
where
    T: H5Type + Copy,
{
    let column_count_u64 =
        u64::try_from(column_count).context("column count did not fit in u64")?;
    let row_count_u64 = u64::try_from(row_count).context("row count did not fit in u64")?;
    let selection = SliceInfo {
        selections: vec![
            SliceInfoElem::Slice {
                start,
                end: start
                    .checked_add(row_count_u64)
                    .context("row selection overflowed")?,
                step: 1,
            },
            SliceInfoElem::Slice {
                start: 0,
                end: column_count_u64,
                step: 1,
            },
        ],
    };
    let rows: ndarray::ArrayD<T> = dataset.read_slice(&selection)?;
    let shape = rows.shape();
    ensure!(
        shape == [row_count, column_count],
        "selected source ground-truth rows had shape {shape:?}, expected [{row_count}, {column_count}]"
    );
    let data = rows
        .as_slice_memory_order()
        .context("selected source ground-truth rows were not contiguous")?;
    let mut output = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let row_start = row
            .checked_mul(column_count)
            .context("selected source row offset overflowed")?;
        let row_end = row_start
            .checked_add(column_count)
            .context("selected source row end overflowed")?;
        output.push(
            data.get(row_start..row_end)
                .context("selected source row was outside dense data")?
                .to_vec(),
        );
    }

    Ok(output)
}

fn read_dense_rows_with_type<T>(
    dataset: &hdf5_reader::Dataset,
    selection: &SliceInfo,
    row_count: usize,
    dimensions: u64,
) -> Result<Vec<Vec<f64>>>
where
    T: H5Type + Copy + Into<f64>,
{
    let rows: ndarray::ArrayD<T> = dataset.read_slice(selection)?;
    let dense_shape = rows.shape();
    ensure!(
        dense_shape.len() == 2,
        "selected dense rows should be 2-D, got shape {dense_shape:?}"
    );
    let selected_rows = dense_shape
        .first()
        .copied()
        .context("selected rows shape did not include row count")?;
    let selected_dimensions = dense_shape
        .get(1)
        .copied()
        .context("selected rows shape did not include dimensions")?;
    ensure!(
        selected_rows == row_count,
        "selected {selected_rows} rows, expected {row_count}"
    );
    ensure!(
        selected_dimensions
            == usize::try_from(dimensions).context("dimensions did not fit in usize")?,
        "selected row dimensions did not match source dimensions"
    );

    let data = rows
        .as_slice_memory_order()
        .context("selected dense rows were not contiguous")?;
    let mut output = Vec::with_capacity(selected_rows);
    for row in 0..selected_rows {
        let row_start = row
            .checked_mul(selected_dimensions)
            .context("selected row offset overflowed")?;
        let row_end = row_start
            .checked_add(selected_dimensions)
            .context("selected row end overflowed")?;
        let row_data = data
            .get(row_start..row_end)
            .context("selected row was outside dense data")?;
        output.push(row_data.iter().map(|value| (*value).into()).collect());
    }

    Ok(output)
}

fn threshold_for(
    expected: &[rig_vector_testkit::ExpectedNeighbor],
    threshold_keep: Option<usize>,
) -> Option<f64> {
    let threshold_keep = threshold_keep?;
    let last_keep_score = expected.get(threshold_keep.checked_sub(1)?)?.score;
    let first_drop_score = expected.get(threshold_keep)?.score;
    Some((last_keep_score + first_drop_score) / 2.0)
}

fn parse_metric(value: &str) -> Result<AnnMetric> {
    match value {
        "cosine" => Ok(AnnMetric::Cosine),
        "l1" => Ok(AnnMetric::L1),
        "l2" => Ok(AnnMetric::L2),
        other => bail!("unsupported metric '{other}', expected cosine, l1, or l2"),
    }
}

fn source_metric_to_ann_metric(
    source_metric: Option<&str>,
    fallback: AnnMetric,
) -> Result<AnnMetric> {
    let Some(source_metric) = source_metric else {
        return Ok(fallback);
    };
    match source_metric {
        "angular" | "cosine" => Ok(AnnMetric::Cosine),
        "euclidean" | "l2" => Ok(AnnMetric::L2),
        "manhattan" | "l1" => Ok(AnnMetric::L1),
        other => bail!(
            "unsupported source metric '{other}', expected angular, cosine, euclidean, l2, manhattan, or l1"
        ),
    }
}

fn parse_usize(value: &str, flag: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("failed to parse {flag} value '{value}' as usize"))
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .with_context(|| format!("{flag} requires a value"))
}

fn print_help() {
    println!(
        "Generate a compact Rig vector-store fixture from a benchmark HDF5 file.\n\
\n\
Usage:\n\
  hdf5_vector_fixture \\\n\
    --input random-xs-20-angular.hdf5 \\\n\
    --output crates/rig-vector-testkit/fixtures/ann/benchmark_derived_ann_random_xs_20_angular_cosine.json \\\n\
    --name benchmark_derived_ann_random_xs_20_angular_cosine \\\n\
    --dataset random-xs-20-angular \\\n\
    --source-kind ann-benchmarks-hdf5 \\\n\
    --url https://ann-benchmarks.com/random-xs-20-angular.hdf5 \\\n\
    --source-metric angular \\\n\
    --metric cosine \\\n\
    --documents 16 \\\n\
    --queries 3 \\\n\
    --top-k 5 \\\n\
    --threshold-keep 3"
    );
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, ensure};

    use super::*;

    #[test]
    fn ground_truth_shape_validation_accepts_matching_shapes() -> Result<()> {
        ensure_ground_truth_shapes((10, 100), (10, 100), 10, 3, 5)?;
        Ok(())
    }

    #[test]
    fn ground_truth_shape_validation_rejects_mismatched_neighbor_distance_shapes() -> Result<()> {
        let result = ensure_ground_truth_shapes((10, 100), (10, 99), 10, 3, 5);

        ensure!(
            result.is_err(),
            "shape validation should reject mismatched /neighbors and /distances shapes"
        );

        Ok(())
    }

    #[test]
    fn ground_truth_shape_validation_rejects_too_few_columns() -> Result<()> {
        let result = ensure_ground_truth_shapes((10, 4), (10, 4), 10, 3, 5);

        ensure!(
            result.is_err(),
            "shape validation should reject source ground truth with too few columns"
        );

        Ok(())
    }

    #[test]
    fn source_metric_mapping_accepts_external_metric_names() -> Result<()> {
        ensure!(source_metric_to_ann_metric(Some("angular"), AnnMetric::L2)? == AnnMetric::Cosine);
        ensure!(
            source_metric_to_ann_metric(Some("euclidean"), AnnMetric::Cosine)? == AnnMetric::L2
        );
        ensure!(
            source_metric_to_ann_metric(Some("manhattan"), AnnMetric::Cosine)? == AnnMetric::L1
        );
        ensure!(source_metric_to_ann_metric(None, AnnMetric::L1)? == AnnMetric::L1);

        Ok(())
    }
}
