# Reproducibility Checklist

## 1. Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Run A Smoke Test

```powershell
python run_quick_test.py
```

Expected generated directories:

```text
results/quick/
models/quick/
```

## 3. Check Included Results

```powershell
python quality_check.py --results-dir results/final --skip-models
```

This checks:

- environment construction,
- final CSV files,
- final figures,
- trade-off summary values.

## 4. Rerun The Full Experiment

Configured full run:

```powershell
python run_experiment.py --output-dir results/full --model-dir models/full
```

Full run without GA baselines:

```powershell
python run_experiment.py --no-ga --output-dir results/full_no_ga --model-dir models/full_no_ga
```

## 5. Expected Final Outputs

```text
training_history.csv
evaluation_results.csv
summary_by_policy.csv
robustness_results.csv
figures/
```

## 6. Result Reading

The key result pattern is:

- higher `alpha` shifts the policy toward communication;
- learned average sensing split decreases as `alpha` increases;
- throughput increases as communication preference increases;
- sensing score should be interpreted as a simulation proxy.
