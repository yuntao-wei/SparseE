# SparseE

## Environment

- Go 1.24 or newer
- Python 3.9 or newer

```bash
git clone https://github.com/yuntao-wei/SparseE.git
cd SparseE
go mod download
```

## Verification

```bash
go test ./...
go test -race ./...
go vet ./...
python3 -m unittest discover -s scripts -p '*_test.py'
```

## Software Benchmark

```bash
go test -v ./benchmarks
go test -run '^$' -bench BenchmarkSparseEProtocol -benchtime=5x -count=3 ./benchmarks
```

## Prediction Script

Run with Markdown output:

```bash
python3 scripts/predict_paper_times.py
```

Select an output format:

```bash
python3 scripts/predict_paper_times.py --format csv
python3 scripts/predict_paper_times.py --format json
```

Use a different software calibration file:

```bash
python3 scripts/predict_paper_times.py \
  --calibration-json path/to/software_calibration.json
```

Override dataset nonzero counts:

```bash
python3 scripts/predict_paper_times.py \
  --nnz-override-json path/to/nnz.json
```

Override the measured seconds per work unit:

```bash
python3 scripts/predict_paper_times.py \
  --seconds-per-work-unit 0.000416932614
```

Show all command-line options:

```bash
python3 scripts/predict_paper_times.py --help
```
