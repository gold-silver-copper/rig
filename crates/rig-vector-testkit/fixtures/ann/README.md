# Vector Fixture Provenance

These compact JSON fixtures are generated from external HDF5 benchmark vector
datasets, but they do not vendor those source HDF5 files and do not claim to
replay the benchmark projects' full ground-truth tasks.

The generator reads a small prefix of each source file's `train` vectors as
fixture documents and a small prefix of `test` vectors as fixture queries. It
then computes exact neighbors over that compact subset with Rig score
conventions:

- cosine: cosine similarity
- l1: negative L1 distance
- l2: negative L2 distance

Fixture validation recomputes that oracle from the committed vectors, so stale
or manually incorrect `expected` values fail before any vector store is queried.

When a source HDF5 file also contains `/neighbors` and `/distances`, the
generator records them as optional `source_ground_truth` on each query. Rig
vector stores return scores rather than raw distances, so those raw source
distances are also converted into Rig's higher-is-better score convention:

- source angular/cosine distance: `1.0 - distance`
- source euclidean/L2 distance: `-distance`
- source manhattan/L1 distance: `-distance`

This source ground truth is provenance and optional recall data. Normal CI still
asserts against the recomputed compact-fixture oracle. The source neighbor IDs
are only populated when the source train row is included in the compact fixture;
otherwise the source row index, raw distance, and converted score are retained
without a fixture document ID. If a fixture is rescored under a different metric
than the source dataset, `source_ground_truth.metric` records the source metric;
the source-ground-truth recall helper rejects those metric mismatches.

Source datasets:

```text
https://ann-benchmarks.com/random-xs-20-angular.hdf5
https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/glove-200-cosine.hdf5
```

The VIBE fixture uses `glove-200-cosine` under its native cosine metric. The
ANN-Benchmarks fixture uses `random-xs-20-angular` as benchmark-shaped vector
data and is rescored under Rig cosine, L1, and L2 semantics.

Regenerate the fixtures with:

```bash
curl -L https://ann-benchmarks.com/random-xs-20-angular.hdf5 -o /tmp/random-xs-20-angular.hdf5
curl -L https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/glove-200-cosine.hdf5 -o /tmp/vibe-glove-200-cosine.hdf5

cargo run -p rig-vector-testkit --features ann-benchmarks --bin hdf5_vector_fixture -- \
  --input /tmp/random-xs-20-angular.hdf5 \
  --output crates/rig-vector-testkit/fixtures/ann/benchmark_derived_ann_random_xs_20_angular_cosine.json \
  --name benchmark_derived_ann_random_xs_20_angular_cosine \
  --dataset random-xs-20-angular \
  --source-kind ann-benchmarks-hdf5 \
  --url https://ann-benchmarks.com/random-xs-20-angular.hdf5 \
  --source-metric angular \
  --metric cosine \
  --documents 16 \
  --queries 3 \
  --top-k 5 \
  --threshold-keep 3

cargo run -p rig-vector-testkit --features ann-benchmarks --bin hdf5_vector_fixture -- \
  --input /tmp/random-xs-20-angular.hdf5 \
  --output crates/rig-vector-testkit/fixtures/ann/benchmark_derived_ann_random_xs_20_angular_l1.json \
  --name benchmark_derived_ann_random_xs_20_angular_l1 \
  --dataset random-xs-20-angular \
  --source-kind ann-benchmarks-hdf5 \
  --url https://ann-benchmarks.com/random-xs-20-angular.hdf5 \
  --source-metric angular \
  --metric l1 \
  --documents 16 \
  --queries 3 \
  --top-k 5 \
  --threshold-keep 3

cargo run -p rig-vector-testkit --features ann-benchmarks --bin hdf5_vector_fixture -- \
  --input /tmp/random-xs-20-angular.hdf5 \
  --output crates/rig-vector-testkit/fixtures/ann/benchmark_derived_ann_random_xs_20_angular_l2.json \
  --name benchmark_derived_ann_random_xs_20_angular_l2 \
  --dataset random-xs-20-angular \
  --source-kind ann-benchmarks-hdf5 \
  --url https://ann-benchmarks.com/random-xs-20-angular.hdf5 \
  --source-metric angular \
  --metric l2 \
  --documents 16 \
  --queries 3 \
  --top-k 5 \
  --threshold-keep 3

cargo run -p rig-vector-testkit --features ann-benchmarks --bin hdf5_vector_fixture -- \
  --input /tmp/vibe-glove-200-cosine.hdf5 \
  --output crates/rig-vector-testkit/fixtures/ann/benchmark_derived_vibe_glove_200_cosine.json \
  --name benchmark_derived_vibe_glove_200_cosine \
  --dataset glove-200-cosine \
  --source-kind vibe-hdf5 \
  --url https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/glove-200-cosine.hdf5 \
  --source-metric cosine \
  --metric cosine \
  --documents 12 \
  --queries 3 \
  --top-k 5 \
  --threshold-keep 3
```
