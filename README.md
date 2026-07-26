# Patient Readmission Analytics

A production-quality healthcare analytics pipeline for analyzing patient readmission rates, ICD diagnosis grouping, cost analysis, and provider performance scoring.

## Architecture

```
Patient-Readmission-Analytics/
├── readmission_analytics/
│   ├── __init__.py
│   ├── run.py                  # CLI entry point
│   ├── requirements.txt
│   ├── configs/
│   │   └── default.yaml        # Pipeline configuration
│   └── src/
│       ├── __init__.py
│       └── pipeline.py          # Core pipeline logic
├── data/                        # SQLite database (auto-created)
├── outputs/                     # CSV analytics outputs
└── README.md
```

**Pipeline Steps:**

1. **Create tables** – patients, admissions, diagnoses, cost_details
2. **Quality checks** – not-null, uniqueness, date ordering validation
3. **Generate data** – synthetic healthcare data (reproducible via seed)
4. **Load** – insert generated DataFrames into SQLite
5. **Readmission analysis** – self-join to flag 30-day readmissions
6. **ICD grouping** – group diagnoses by ICD-10 chapter
7. **Cost analysis** – aggregate costs with outlier removal (percentile cap)
8. **Provider scoring** – composite score from readmission rate, cost, LOS
9. **Save outputs** – export results as CSV files

## Self-Join Explanation

The readmission flag uses a **correlated subquery** pattern on the admissions table. For each admission `a`, it checks whether the same patient has another admission `b` where:

```
b.admission_date > a.discharge_date
AND b.admission_date <= date(a.discharge_date, '+30 days')
```

This is equivalent to a self-join but more efficient for the "exists" semantics. The query runs per-row via a correlated `EXISTS` subquery, and a second correlated subquery computes `days_to_readmit` as the gap to the earliest subsequent admission.

## ICD Grouping

Diagnoses are grouped by **ICD-10 chapter**. The first character of each ICD code maps to one of 26 chapters (A–Z). For example:

| Chapter | Range       | Description                                  |
|---------|-------------|----------------------------------------------|
| I       | I00 – I99   | Diseases of the circulatory system           |
| J       | J00 – J99   | Diseases of the respiratory system           |
| E       | E00 – E89   | Endocrine, nutritional and metabolic diseases|

## PERCENTILE_CONT

The provider scoring step attempts to use PostgreSQL's `PERCENTILE_CONT(0.5)` window function to compute median-normalised metrics. When running on SQLite (which lacks this function), the pipeline automatically falls back to Python-based normalisation via `numpy`/`pandas`.

The composite provider score is:

```
score = 0.4 × normalised_readmission_rate
      + 0.3 × normalised_avg_cost
      + 0.3 × normalised_avg_los
```

Lower scores indicate better performance (lower readmission rate, lower cost, shorter length of stay).

## Quick Start

```bash
# Install dependencies
pip install -r readmission_analytics/requirements.txt

# Run full pipeline (generate → analyze → save)
python -m readmission_analytics.run full

# Or run individual steps
python -m readmission_analytics.run generate   # create data only
python -m readmission_analytics.run quality    # run quality checks
python -m readmission_analytics.run analyze    # analytics on existing data

# Specify a custom config
python -m readmission_analytics.run full --config my_config.yaml

# Enable debug logging
python -m readmission_analytics.run full --log-level DEBUG
```

## Output Files

After a full run, the `outputs/` directory contains:

| File                  | Description                                         |
|-----------------------|-----------------------------------------------------|
| `readmissions.csv`    | Per-admission readmission flags and days-to-readmit |
| `icd_groups.csv`      | Diagnosis counts by ICD-10 chapter                  |
| `cost_analysis.csv`   | Aggregate cost statistics and category breakdown    |
| `provider_scores.csv` | Provider rankings by composite performance score    |

## Configuration

Edit `readmission_analytics/configs/default.yaml` to adjust:

- **data**: Patient/admission/provider counts, random seed
- **quality_checks**: Not-null and uniqueness constraints per table
- **readmission**: Window size (days), minimum admissions threshold
- **icd**: ICD version and grouping method
- **cost**: Outlier percentile for cost capping
- **provider**: Minimum patients per provider, scoring weights
- **output**: Output directory, CSV export toggle

## License

MIT
