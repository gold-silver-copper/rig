# ANN Fixture Provenance

These compact JSON fixtures are generated from the ANN-Benchmarks
`random-xs-20-angular.hdf5` dataset:

```text
https://ann-benchmarks.com/random-xs-20-angular.hdf5
```

The generator reads the first 16 `train` vectors as fixture documents and the
first 3 `test` vectors as fixture queries. It then computes expected neighbors
for Rig's score conventions over that compact subset:

- cosine: cosine similarity
- l1: negative L1 distance
- l2: negative L2 distance

The source HDF5 file is not committed. Regenerate the fixtures with:

```bash
curl -L https://ann-benchmarks.com/random-xs-20-angular.hdf5 -o /tmp/random-xs-20-angular.hdf5

cargo run -p rig-vector-testkit --features ann-benchmarks --bin ann_benchmarks_fixture -- \
  --input /tmp/random-xs-20-angular.hdf5 \
  --output crates/rig-vector-testkit/fixtures/ann/ann_benchmarks_random_xs_20_angular_cosine.json \
  --name ann_benchmarks_random_xs_20_angular_cosine \
  --dataset random-xs-20-angular \
  --url https://ann-benchmarks.com/random-xs-20-angular.hdf5 \
  --source-metric angular \
  --metric cosine \
  --documents 16 \
  --queries 3 \
  --top-k 5 \
  --threshold-keep 3

cargo run -p rig-vector-testkit --features ann-benchmarks --bin ann_benchmarks_fixture -- \
  --input /tmp/random-xs-20-angular.hdf5 \
  --output crates/rig-vector-testkit/fixtures/ann/ann_benchmarks_random_xs_20_angular_l1.json \
  --name ann_benchmarks_random_xs_20_angular_l1 \
  --dataset random-xs-20-angular \
  --url https://ann-benchmarks.com/random-xs-20-angular.hdf5 \
  --source-metric angular \
  --metric l1 \
  --documents 16 \
  --queries 3 \
  --top-k 5 \
  --threshold-keep 3

cargo run -p rig-vector-testkit --features ann-benchmarks --bin ann_benchmarks_fixture -- \
  --input /tmp/random-xs-20-angular.hdf5 \
  --output crates/rig-vector-testkit/fixtures/ann/ann_benchmarks_random_xs_20_angular_l2.json \
  --name ann_benchmarks_random_xs_20_angular_l2 \
  --dataset random-xs-20-angular \
  --url https://ann-benchmarks.com/random-xs-20-angular.hdf5 \
  --source-metric angular \
  --metric l2 \
  --documents 16 \
  --queries 3 \
  --top-k 5 \
  --threshold-keep 3
```
