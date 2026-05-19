use std::env;
use std::fs::File;
use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use hdf5_reader::{H5Type, Hdf5File, SliceInfo, SliceInfoElem};
use rig_vector_testkit::{
    AnnFixture, AnnFixtureSource, AnnMetric, FixtureDocument, FixtureQuery,
    compute_expected_neighbors,
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

    let train_vectors =
        read_dense_rows(&train, 0, args.documents).context("failed to read train rows")?;
    let query_vectors =
        read_dense_rows(&test, 0, args.queries).context("failed to read test rows")?;
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
        .enumerate()
        .map(|(index, vector)| FixtureDocument {
            id: format!("{source_prefix}:{}:train:{index}", args.dataset),
            text: format!("{source_prefix}:{}:train:{index}", args.dataset),
            vector: vector.clone(),
        })
        .collect::<Vec<_>>();

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
            train_count: args.documents,
            test_start: 0,
            test_count: args.queries,
            top_k: args.top_k,
            generated_by: "cargo run -p rig-vector-testkit --features ann-benchmarks --bin ann_benchmarks_fixture".to_string(),
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
    let shape = dataset.shape();
    ensure!(
        shape.len() == 2,
        "expected a dense 2-D dataset, got shape {shape:?}"
    );
    let total_rows = shape
        .first()
        .copied()
        .context("dense dataset shape did not include row count")?;
    let dimensions = shape
        .get(1)
        .copied()
        .context("dense dataset shape did not include dimensions")?;
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
  ann_benchmarks_fixture \\\n\
    --input random-xs-20-angular.hdf5 \\\n\
    --output crates/rig-vector-testkit/fixtures/ann/ann_benchmarks_random_xs_20_angular_cosine.json \\\n\
    --name ann_benchmarks_random_xs_20_angular_cosine \\\n\
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
