"""Patient Readmission Analytics Pipeline.

Orchestrates the full analytics workflow:
create tables -> quality checks -> generate data -> load -> readmission analysis
-> ICD grouping -> cost analysis -> provider scoring -> save outputs.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ICD-10 chapter mapping
# ---------------------------------------------------------------------------
ICD10_CHAPTERS: dict[str, tuple[str, str]] = {
    "A": ("A00", "B99"), "B": ("B00", "B99"),
    "C": ("C00", "D49"), "D": ("D50", "D89"),
    "E": ("E00", "E89"), "F": ("F01", "F99"),
    "G": ("G00", "G99"), "H": ("H00", "H59"),
    "I": ("I00", "I99"), "J": ("J00", "J99"),
    "K": ("K00", "K95"), "L": ("L00", "L99"),
    "M": ("M00", "M99"), "N": ("N00", "N99"),
    "O": ("O00", "O9A"), "P": ("P00", "P96"),
    "Q": ("Q00", "Q99"), "R": ("R00", "R99"),
    "S": ("S00", "T88"), "T": ("T00", "T88"),
    "U": ("U00", "U99"), "V": ("V00", "V99"),
    "W": ("W00", "W99"), "X": ("X00", "X99"),
    "Y": ("Y00", "Y99"), "Z": ("Z00", "Z99"),
}


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Loaded config from %s", config_path)
    return cfg


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------
def create_tables(engine: Engine) -> None:
    """Create the SQLite schema for patients, admissions, diagnoses, and costs.

    Args:
        engine: SQLAlchemy engine instance.
    """
    statements = [
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id   INTEGER PRIMARY KEY,
            mrn          TEXT    UNIQUE NOT NULL,
            name         TEXT    NOT NULL,
            date_of_birth DATE,
            gender       TEXT,
            race         TEXT,
            insurance    TEXT,
            zip_code     TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS admissions (
            admission_id   INTEGER PRIMARY KEY,
            patient_id     INTEGER NOT NULL,
            admission_date TEXT    NOT NULL,
            discharge_date TEXT,
            provider_id    INTEGER,
            primary_diagnosis TEXT,
            total_cost     REAL,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS diagnoses (
            diagnosis_id   INTEGER PRIMARY KEY,
            admission_id   INTEGER NOT NULL,
            icd_code       TEXT    NOT NULL,
            description    TEXT,
            FOREIGN KEY (admission_id) REFERENCES admissions(admission_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cost_details (
            cost_id        INTEGER PRIMARY KEY,
            admission_id   INTEGER NOT NULL,
            cost_category  TEXT    NOT NULL,
            amount         REAL    NOT NULL,
            FOREIGN KEY (admission_id) REFERENCES admissions(admission_id)
        )
        """,
    ]
    with engine.connect() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
        conn.commit()
    logger.info("Database tables created successfully")


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def generate_data(cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Generate synthetic healthcare data.

    Args:
        cfg: Configuration dictionary containing data generation parameters.
        rng: NumPy random generator for reproducibility.

    Returns:
        Dictionary mapping table names to DataFrames.
    """
    n_patients = cfg["data"]["n_patients"]
    n_admissions = cfg["data"]["n_admissions"]
    n_providers = cfg["data"]["n_providers"]

    logger.info(
        "Generating data: %d patients, %d admissions, %d providers",
        n_patients, n_admissions, n_providers,
    )

    # --- patients ---
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer",
                   "Michael", "Linda", "William", "Elizabeth", "David", "Barbara",
                   "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
                   "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy",
                   "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                  "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez",
                  "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor",
                  "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
                  "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"]
    genders = ["M", "F"]
    races = ["White", "Black", "Hispanic", "Asian", "Other"]
    insurances = ["Medicare", "Medicaid", "Private", "Self-Pay"]
    zip_codes = [f"{i:05d}" for i in range(10000, 10100)]

    patients = pd.DataFrame({
        "patient_id": range(1, n_patients + 1),
        "mrn": [f"MRN{str(i).zfill(6)}" for i in range(1, n_patients + 1)],
        "name": [
            f"{rng.choice(first_names)} {rng.choice(last_names)}"
            for _ in range(n_patients)
        ],
        "date_of_birth": pd.date_range("1930-01-01", periods=n_patients, freq="5D").strftime("%Y-%m-%d"),
        "gender": rng.choice(genders, n_patients).tolist(),
        "race": rng.choice(races, n_patients, p=[0.5, 0.2, 0.15, 0.1, 0.05]).tolist(),
        "insurance": rng.choice(insurances, n_patients, p=[0.4, 0.25, 0.25, 0.1]).tolist(),
        "zip_code": rng.choice(zip_codes, n_patients).tolist(),
    })

    # --- admissions ---
    base_date = pd.Timestamp("2022-01-01")
    admission_dates = base_date + pd.to_timedelta(
        rng.integers(0, 730, size=n_admissions), unit="D"
    )
    los_days = rng.exponential(scale=5, size=n_admissions).clip(1, 90).astype(int)
    discharge_dates = admission_dates + pd.to_timedelta(los_days, unit="D")

    admissions = pd.DataFrame({
        "admission_id": range(1, n_admissions + 1),
        "patient_id": rng.integers(1, n_patients + 1, size=n_admissions).tolist(),
        "admission_date": admission_dates.strftime("%Y-%m-%d").tolist(),
        "discharge_date": discharge_dates.strftime("%Y-%m-%d").tolist(),
        "provider_id": rng.integers(1, n_providers + 1, size=n_admissions).tolist(),
        "primary_diagnosis": [
            f"{rng.choice(list(ICD10_CHAPTERS.keys()))}{rng.integers(0, 999):03d}"
            for _ in range(n_admissions)
        ],
        "total_cost": (
            rng.lognormal(mean=9.5, sigma=0.8, size=n_admissions).clip(500, 200000).round(2)
        ).tolist(),
    })

    # --- diagnoses ---
    icd_codes = [
        "A09", "B34", "C34", "D50", "E11", "F32", "G40", "I10",
        "I21", "I50", "J18", "K80", "L03", "M54", "N17", "O80",
        "P07", "Q21", "R07", "S72", "T78", "V29", "W01", "X59", "Y84", "Z00",
    ]
    diagnosis_descriptions = {
        "A09": "Infectious gastroenteritis", "B34": "Unspecified viral infection",
        "C34": "Lung malignancy", "D50": "Iron deficiency anaemia",
        "E11": "Type 2 diabetes mellitus", "F32": "Major depressive disorder",
        "G40": "Epilepsy", "I10": "Essential hypertension",
        "I21": "Acute myocardial infarction", "I50": "Heart failure",
        "J18": "Pneumonia", "K80": "Cholelithiasis",
        "L03": "Cellulitis", "M54": "Low back pain",
        "N17": "Acute kidney injury", "O80": "Normal delivery",
        "P07": "Low birth weight", "Q21": "Ventricular septal defect",
        "R07": "Chest pain", "S72": "Femoral fracture",
        "T78": "Adverse effect NEC", "V29": "Pedestrian conveyance NEC",
        "W01": "Fall on/from stairs", "X59": "Exposure to NEC",
        "Y84": "Other medical procedures", "Z00": "General exam",
    }

    n_dx = rng.integers(1, 4, size=n_admissions)
    diagnosis_rows = []
    dx_id = 1
    for adm_id, count in zip(admissions["admission_id"], n_dx):
        for _ in range(int(count)):
            code = rng.choice(icd_codes)
            diagnosis_rows.append({
                "diagnosis_id": dx_id,
                "admission_id": adm_id,
                "icd_code": code,
                "description": diagnosis_descriptions.get(code, "Unspecified"),
            })
            dx_id += 1
    diagnoses = pd.DataFrame(diagnosis_rows)

    # --- cost_details (vectorized) ---
    categories = np.array(["Room", "Medication", "Procedure", "Lab", "Imaging", "Other"])
    cats_per_admission = rng.integers(2, len(categories) + 1, size=n_admissions)
    total_rows = int(cats_per_admission.sum())
    adm_ids_rep = np.repeat(admissions["admission_id"].values, cats_per_admission)
    total_costs_rep = np.repeat(admissions["total_cost"].values, cats_per_admission)
    # build category choices per admission row
    cat_choices = np.empty(total_rows, dtype=object)
    idx = 0
    for n in cats_per_admission:
        cat_choices[idx:idx + n] = rng.choice(categories, size=int(n), replace=False)
        idx += int(n)
    # split costs via dirichlet
    alpha = np.ones(6)
    amounts = np.empty(total_rows)
    idx = 0
    for n in cats_per_admission:
        amounts[idx:idx + int(n)] = rng.dirichlet(alpha[:int(n)]) * total_costs_rep[idx]
        idx += int(n)
    cost_details = pd.DataFrame({
        "cost_id": range(1, total_rows + 1),
        "admission_id": adm_ids_rep.tolist(),
        "cost_category": cat_choices.tolist(),
        "amount": amounts.round(2).tolist(),
    })

    logger.info("Synthetic data generation complete")
    return {
        "patients": patients,
        "admissions": admissions,
        "diagnoses": diagnoses,
        "cost_details": cost_details,
    }


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def run_quality_checks(
    engine: Engine, cfg: dict[str, Any]
) -> dict[str, list[str]]:
    """Run data quality checks against the configured rules.

    Checks include not-null constraints, uniqueness, and date ordering.

    Args:
        engine: SQLAlchemy engine with loaded tables.
        cfg: Configuration dictionary with quality check rules.

    Returns:
        Dictionary of check_name -> list of error messages (empty if passed).
    """
    qc_cfg = cfg.get("quality_checks", {})
    results: dict[str, list[str]] = {}

    with engine.connect() as conn:
        for tbl, rules in qc_cfg.items():
            if not isinstance(rules, dict):
                continue
            not_null_cols = rules.get("not_null", [])
            for col in not_null_cols:
                check_name = f"{tbl}.{col}.not_null"
                df = pd.read_sql(
                    text(f"SELECT COUNT(*) AS cnt FROM {tbl} WHERE {col} IS NULL"),
                    conn,
                )
                null_count = int(df["cnt"].iloc[0])
                if null_count > 0:
                    results.setdefault(check_name, []).append(
                        f"{null_count} NULL values in {tbl}.{col}"
                    )

            # uniqueness checks
            for col in rules.get("unique", []):
                check_name = f"{tbl}.{col}.unique"
                df = pd.read_sql(
                    text(f"SELECT {col}, COUNT(*) AS cnt FROM {tbl} GROUP BY {col} HAVING cnt > 1"),
                    conn,
                )
                if not df.empty:
                    results.setdefault(check_name, []).append(
                        f"{len(df)} duplicate {col} values in {tbl}"
                    )

            # date ordering checks
            date_cols = rules.get("date_order", [])
            if len(date_cols) == 2:
                col_a, col_b = date_cols
                check_name = f"{tbl}.{col_a}_le_{col_b}"
                df = pd.read_sql(
                    text(
                        f"SELECT COUNT(*) AS cnt FROM {tbl} "
                        f"WHERE {col_a} IS NOT NULL AND {col_b} IS NOT NULL "
                        f"AND {col_a} > {col_b}"
                    ),
                    conn,
                )
                bad_count = int(df["cnt"].iloc[0])
                if bad_count > 0:
                    results.setdefault(check_name, []).append(
                        f"{bad_count} rows where {col_a} > {col_b} in {tbl}"
                    )

    if not results:
        logger.info("All quality checks passed")
    else:
        logger.warning("Quality check issues found: %d rules failed", len(results))
    return results


# ---------------------------------------------------------------------------
# Readmission analysis (self-join)
# ---------------------------------------------------------------------------
def compute_readmissions(
    engine: Engine, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Compute readmission flags using a vectorised pandas self-join.

    Loads admissions into a DataFrame, performs a merge on patient_id where
    the next admission is after the current discharge, then keeps only the
    earliest next admission per discharge.  This avoids the O(n^2) cost of
    correlated subqueries in SQLite.

    Args:
        engine: SQLAlchemy engine with admissions table loaded.
        cfg: Configuration with readmission.window_days.

    Returns:
        DataFrame with readmission flags per admission.
    """
    window = cfg["readmission"]["window_days"]

    adm = pd.read_sql("SELECT admission_id, patient_id, admission_date, discharge_date, total_cost FROM admissions", engine)
    adm["admission_date"] = pd.to_datetime(adm["admission_date"])
    adm["discharge_date"] = pd.to_datetime(adm["discharge_date"])
    adm = adm.sort_values(["patient_id", "admission_date"]).reset_index(drop=True)

    # Self-join: find all subsequent admissions for the same patient
    next_adm = adm[["admission_id", "patient_id", "admission_date"]].rename(
        columns={"admission_id": "next_admission_id", "admission_date": "next_admission_date"}
    )
    merged = adm.merge(next_adm, on="patient_id")
    merged = merged[merged["next_admission_date"] > merged["discharge_date"]]

    # Keep only the earliest next admission per current admission
    merged = merged.sort_values(["admission_id", "next_admission_date"])
    earliest = merged.groupby("admission_id").first().reset_index()
    earliest["days_to_readmit"] = (
        earliest["next_admission_date"] - earliest["discharge_date"]
    ).dt.days

    result = adm.copy()
    result = result.merge(
        earliest[["admission_id", "next_admission_date", "days_to_readmit"]],
        on="admission_id", how="left",
    )
    result["readmitted"] = result["days_to_readmit"].notna() & (result["days_to_readmit"] <= window)
    result["readmitted"] = result["readmitted"].astype(int)

    readmitted_count = int(result["readmitted"].sum())
    total = len(result)
    logger.info(
        "Readmission analysis complete: %d / %d (%.1f%%) readmitted within %d days",
        readmitted_count, total,
        100 * readmitted_count / total if total else 0,
        window,
    )
    return result


# ---------------------------------------------------------------------------
# ICD chapter grouping
# ---------------------------------------------------------------------------
def group_icd_by_chapter(engine: Engine) -> pd.DataFrame:
    """Group diagnosis records by ICD-10 chapter.

    Extracts the leading letter from each ICD code and maps it to a
    chapter name.

    Args:
        engine: SQLAlchemy engine with diagnoses table.

    Returns:
        DataFrame with columns: icd_chapter, chapter_name, diagnosis_count.
    """
    query = text("""
        SELECT
            UPPER(SUBSTR(d.icd_code, 1, 1)) AS icd_chapter,
            COUNT(DISTINCT d.diagnosis_id)   AS diagnosis_count
        FROM diagnoses d
        GROUP BY icd_chapter
        ORDER BY diagnosis_count DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    chapter_names = {
        "A": "Certain infectious and parasitic diseases",
        "B": "Certain infectious and parasitic diseases",
        "C": "Neoplasms",
        "D": "Diseases of the blood",
        "E": "Endocrine, nutritional and metabolic diseases",
        "F": "Mental, behavioural and neurodevelopmental disorders",
        "G": "Diseases of the nervous system",
        "H": "Diseases of the eye and adnexa / ear",
        "I": "Diseases of the circulatory system",
        "J": "Diseases of the respiratory system",
        "K": "Diseases of the digestive system",
        "L": "Diseases of the skin and subcutaneous tissue",
        "M": "Diseases of the musculoskeletal system",
        "N": "Diseases of the genitourinary system",
        "O": "Pregnancy, childbirth and puerperium",
        "P": "Certain conditions originating in the perinatal period",
        "Q": "Congenital malformations",
        "R": "Symptoms, signs and abnormal clinical findings",
        "S": "Injury, poisoning and certain other consequences",
        "T": "Injury, poisoning and certain other consequences",
        "U": "Codes for special purposes",
        "V": "External causes of morbidity",
        "W": "External causes of morbidity",
        "X": "External causes of morbidity",
        "Y": "External causes of morbidity",
        "Z": "Factors influencing health status",
    }
    df["chapter_name"] = df["icd_chapter"].map(chapter_names).fillna("Unknown")
    logger.info("ICD grouping complete: %d chapters identified", len(df))
    return df


# ---------------------------------------------------------------------------
# Cost analysis with outlier removal
# ---------------------------------------------------------------------------
def compute_cost_analysis(
    engine: Engine, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Compute per-admission cost statistics with outlier removal.

    Outliers above the configured percentile are excluded before
    aggregation.

    Args:
        engine: SQLAlchemy engine with admissions and cost_details tables.
        cfg: Configuration with cost.outlier_percentile.

    Returns:
        DataFrame with cost summary per admission.
    """
    percentile = cfg["cost"]["outlier_percentile"]

    with engine.connect() as conn:
        costs_df = pd.read_sql("SELECT admission_id, total_cost FROM admissions", conn)
        detail_df = pd.read_sql("SELECT admission_id, cost_category, amount FROM cost_details", conn)

    # remove outliers
    threshold = np.percentile(costs_df["total_cost"], percentile)
    costs_df = costs_df[costs_df["total_cost"] <= threshold].copy()
    logger.info(
        "Cost analysis: removed outliers above P%d (threshold $%.2f), %d admissions remain",
        percentile, threshold, len(costs_df),
    )

    # category breakdown
    cat_summary = (
        detail_df.groupby("cost_category")["amount"]
        .agg(["sum", "mean", "count"])
        .round(2)
    )
    cat_summary.columns = ["total_amount", "mean_amount", "line_count"]

    # aggregate stats
    stats = {
        "total_cost_mean": costs_df["total_cost"].mean(),
        "total_cost_median": costs_df["total_cost"].median(),
        "total_cost_std": costs_df["total_cost"].std(),
        "total_cost_min": costs_df["total_cost"].min(),
        "total_cost_max": costs_df["total_cost"].max(),
        "n_admissions": len(costs_df),
    }
    summary_df = pd.DataFrame([stats]).round(2)

    logger.info(
        "Cost summary: mean=$%.2f, median=$%.2f, std=$%.2f",
        stats["total_cost_mean"], stats["total_cost_median"], stats["total_cost_std"],
    )
    return pd.concat([
        summary_df,
        cat_summary.reset_index(),
    ], ignore_index=True)


# ---------------------------------------------------------------------------
# Provider performance scoring
# ---------------------------------------------------------------------------
def compute_provider_scores(
    engine: Engine, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Compute provider performance scores.

    Score = w1 * norm_readmission_rate + w2 * norm_avg_cost + w3 * norm_avg_los,
    where lower readmission rate, lower cost, and shorter LOS are better.

    PERCENTILE_CONT is used to normalize metrics to 0-1 range so they are
    directly comparable across providers.

    Args:
        engine: SQLAlchemy engine.
        cfg: Configuration with provider.min_patients and weights.

    Returns:
        DataFrame with provider scores ranked by composite score.
    """
    min_patients = cfg["provider"]["min_patients"]
    weights = cfg["provider"]["weights"]

    query = text(f"""
        WITH provider_stats AS (
            SELECT
                a.provider_id,
                COUNT(DISTINCT a.patient_id)              AS n_patients,
                AVG(CASE WHEN r.readmitted = 1 THEN 1.0 ELSE 0.0 END) AS readmission_rate,
                AVG(a.total_cost)                          AS avg_cost,
                AVG(julianday(a.discharge_date) - julianday(a.admission_date)) AS avg_los
            FROM admissions a
            LEFT JOIN (
                SELECT a2.admission_id,
                       CASE WHEN EXISTS (
                           SELECT 1 FROM admissions b
                           WHERE b.patient_id = a2.patient_id
                             AND b.admission_date > a2.discharge_date
                             AND date(b.admission_date) <= date(a2.discharge_date, '+{cfg['readmission']['window_days']} days')
                       ) THEN 1 ELSE 0 END AS readmitted
                FROM admissions a2
            ) r ON a.admission_id = r.admission_id
            GROUP BY a.provider_id
            HAVING n_patients >= {min_patients}
        ),
        percentiles AS (
            SELECT
                readmission_rate,
                avg_cost,
                avg_los,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY readmission_rate) AS med_rr,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_cost)         AS med_cost,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_los)          AS med_los
            FROM provider_stats
        )
        SELECT
            ps.provider_id,
            ps.n_patients,
            ROUND(ps.readmission_rate, 4) AS readmission_rate,
            ROUND(ps.avg_cost, 2)         AS avg_cost,
            ROUND(ps.avg_los, 2)          AS avg_los_days
        FROM provider_stats ps
        ORDER BY ps.provider_id
    """)

    with engine.connect() as conn:
        try:
            df = pd.read_sql(query, conn)
        except Exception:
            # SQLite does not support PERCENTILE_CONT; fall back to Python
            logger.warning("PERCENTILE_CONT unavailable in SQLite, using Python fallback")
            df = _provider_score_fallback(engine, cfg)

    if df.empty:
        logger.warning("No providers meet minimum patient threshold (%d)", min_patients)
        return df

    # min-max normalisation (lower is better for all three)
    for col in ["readmission_rate", "avg_cost", "avg_los_days"]:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df[f"norm_{col}"] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f"norm_{col}"] = 0.0

    df["composite_score"] = (
        weights["readmission_rate"] * df["norm_readmission_rate"]
        + weights["avg_cost"] * df["norm_avg_cost"]
        + weights["avg_los"] * df["norm_avg_los_days"]
    ).round(4)

    df["rank"] = df["composite_score"].rank(ascending=True, method="min").astype(int)
    df = df.sort_values("rank")

    logger.info("Provider scoring complete: %d providers scored", len(df))
    return df


def _provider_score_fallback(
    engine: Engine, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Fallback provider scoring when PERCENTILE_CONT is unavailable.

    Computes provider stats in Python instead of SQL.

    Args:
        engine: SQLAlchemy engine.
        cfg: Configuration dictionary.

    Returns:
        DataFrame with provider statistics.
    """
    window = cfg["readmission"]["window_days"]

    admissions_df = pd.read_sql("SELECT * FROM admissions", engine)

    readmit_query = text(f"""
        SELECT a.admission_id,
               CASE WHEN EXISTS (
                   SELECT 1 FROM admissions b
                   WHERE b.patient_id = a.patient_id
                     AND b.admission_date > a.discharge_date
                     AND date(b.admission_date) <= date(a.discharge_date, '+{window} days')
               ) THEN 1 ELSE 0 END AS readmitted
        FROM admissions a
    """)
    with engine.connect() as conn:
        readmit_df = pd.read_sql(readmit_query, conn)

    merged = admissions_df.merge(readmit_df, on="admission_id", how="left")
    merged["los"] = (
        pd.to_datetime(merged["discharge_date"]) - pd.to_datetime(merged["admission_date"])
    ).dt.days

    stats = (
        merged.groupby("provider_id")
        .agg(
            n_patients=("patient_id", "nunique"),
            readmission_rate=("readmitted", "mean"),
            avg_cost=("total_cost", "mean"),
            avg_los=("los", "mean"),
        )
        .reset_index()
    )
    stats = stats[stats["n_patients"] >= cfg["provider"]["min_patients"]]
    stats = stats.rename(columns={"avg_los": "avg_los_days"}).round(4)
    return stats


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------
def save_outputs(
    results: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> Path:
    """Persist pipeline outputs to CSV files.

    Args:
        results: Dictionary mapping output names to DataFrames.
        cfg: Configuration with output.output_dir and output.save_csv.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(cfg["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in results.items():
        if cfg["output"].get("save_csv", True) and isinstance(df, pd.DataFrame) and not df.empty:
            out_path = output_dir / f"{name}.csv"
            df.to_csv(out_path, index=False)
            logger.info("Saved %s -> %s (%d rows)", name, out_path, len(df))

    return output_dir


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------
class ReadmissionPipeline:
    """End-to-end readmission analytics pipeline.

    Steps:
        1. Create database tables
        2. Run quality checks
        3. Generate synthetic data
        4. Load data into SQLite
        5. Compute readmission flags (self-join)
        6. Group ICD codes by chapter
        7. Analyse costs (outlier removal)
        8. Score providers
        9. Save outputs
    """

    def __init__(self, config_path: str | Path = "configs/default.yaml") -> None:
        """Initialise the pipeline.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.cfg = load_config(config_path)
        self.engine: Engine = create_engine(
            self.cfg["database"]["url"], echo=False
        )
        self._rng = np.random.default_rng(self.cfg["data"]["seed"])

    def create(self) -> None:
        """Create database tables."""
        create_tables(self.engine)

    def quality(self) -> dict[str, list[str]]:
        """Run data quality checks.

        Returns:
            Dictionary of check_name -> list of error messages.
        """
        return run_quality_checks(self.engine, self.cfg)

    def generate(self) -> dict[str, pd.DataFrame]:
        """Generate synthetic data and load into database.

        Returns:
            Dictionary of table_name -> DataFrame.
        """
        data = generate_data(self.cfg, self._rng)
        for table_name, df in data.items():
            df.to_sql(table_name, self.engine, if_exists="replace", index=False)
            logger.info("Loaded %s: %d rows", table_name, len(df))
        return data

    def analyze(self) -> dict[str, pd.DataFrame]:
        """Run the full analytics suite.

        Returns:
            Dictionary of analysis_name -> DataFrame.
        """
        readmissions = compute_readmissions(self.engine, self.cfg)
        icd_groups = group_icd_by_chapter(self.engine)
        cost_analysis = compute_cost_analysis(self.engine, self.cfg)
        provider_scores = compute_provider_scores(self.engine, self.cfg)

        return {
            "readmissions": readmissions,
            "icd_groups": icd_groups,
            "cost_analysis": cost_analysis,
            "provider_scores": provider_scores,
        }

    def run(self, mode: str = "full") -> dict[str, Any]:
        """Execute the pipeline.

        Args:
            mode: One of 'full', 'generate', 'analyze', 'quality'.

        Returns:
            Dictionary with pipeline results keyed by analysis name.
        """
        logger.info("Starting pipeline in '%s' mode", mode)
        results: dict[str, Any] = {}

        if mode in ("full", "generate"):
            self.create()
            self.generate()

        if mode in ("full", "quality"):
            results["quality"] = self.quality()

        if mode in ("full", "analyze"):
            results.update(self.analyze())

        if mode == "full":
            save_outputs(results, self.cfg)

        logger.info("Pipeline '%s' finished successfully", mode)
        return results
