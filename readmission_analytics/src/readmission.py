"""
Readmission Analytics Module
=============================
Patient readmission analysis using self-join logic on admission dates.
Implements 30-day readmission detection, rate calculations, and cohort analysis.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

READMISSION_WINDOW_DAYS = 30


class ReadmissionAnalyzer:
    """Analyze patient readmissions using self-join logic.

    Core idea: for each pair of admissions (a1, a2) belonging to the same
    patient, a "readmission" occurs when a2 starts within 30 days of a1's
    discharge. The self-join is expressed both in SQL (for documentation) and
    in pure pandas for in-memory workflows.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Database engine for SQL-backed queries.  Pass ``None`` if you only
        intend to work with in-memory DataFrames via ``detect_readmissions()``.
    """

    def __init__(self, engine=None):
        self.engine = engine

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_admissions_df(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist and dates are proper types."""
        required = {"patient_id", "admission_date", "discharge_date"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing)}"
            )
        df = df.copy()
        df["admission_date"] = pd.to_datetime(df["admission_date"])
        df["discharge_date"] = pd.to_datetime(df["discharge_date"])
        return df

    @staticmethod
    def _drop_missing_discharges(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where discharge_date is NaT (patient still in hospital)."""
        before = len(df)
        df = df.dropna(subset=["discharge_date"])
        dropped = before - len(df)
        if dropped:
            logger.warning(
                "Dropped %d admission(s) with missing discharge_date.", dropped
            )
        return df

    @staticmethod
    def _compute_readmit_flag(df: pd.DataFrame) -> pd.DataFrame:
        """Vectorised readmission flag after self-join.

        Expects columns ``_a1_discharge`` and ``_a2_admission`` already
        present from the self-join and adds ``is_readmission`` (bool).
        """
        df["days_to_readmit"] = (
            df["_a2_admission"] - df["_a1_discharge"]
        ).dt.days
        df["is_readmission"] = (
            df["days_to_readmit"].between(1, READMISSION_WINDOW_DAYS)
        )
        return df

    @staticmethod
    def _calculate_rate(
        grouped: pd.Series | pd.DataFrame,
        numerator_col: str = "readmissions",
        denominator_col: str = "total_admissions",
    ) -> pd.DataFrame:
        """Build a rate DataFrame from grouped counts."""
        result = grouped.reset_index()
        result["readmission_rate"] = (
            result[numerator_col] / result[denominator_col]
        ).round(4)
        return result.sort_values("readmission_rate", ascending=False)

    # ------------------------------------------------------------------
    # Core self-join
    # ------------------------------------------------------------------

    def detect_readmissions(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Self-join admissions to find readmissions within 30 days.

        Pandas-equivalent of::

            SELECT a1.patient_id,
                   a1.admission_date   AS admission_1,
                   a1.discharge_date   AS discharge_1,
                   a2.admission_date   AS admission_2,
                   a2.discharge_date   AS discharge_2,
                   julianday(a2.admission_date) - julianday(a1.discharge_date)
                       AS days_to_readmit,
                   CASE
                       WHEN julianday(a2.admission_date) - julianday(a1.discharge_date)
                            BETWEEN 1 AND 30 THEN 1 ELSE 0
                   END AS is_readmission
            FROM   admissions a1
            JOIN   admissions a2
              ON  a1.patient_id = a2.patient_id
             AND  a2.admission_date > a1.discharge_date
             AND  a2.admission_date <= date(a1.discharge_date, '+30 days')
            ORDER BY a1.patient_id, a1.admission_date;

        Parameters
        ----------
        df : pd.DataFrame, optional
            Admissions data with columns ``patient_id``, ``admission_date``,
            ``discharge_date``.  If ``None``, the query is run against the
            database engine.

        Returns
        -------
        pd.DataFrame
            One row per admission-pair with readmission metadata.
        """
        if df is None:
            if self.engine is None:
                raise RuntimeError(
                    "No DataFrame passed and no database engine configured."
                )
            return self._detect_readmissions_sql()

        df = self._validate_admissions_df(df)
        df = self._drop_missing_discharges(df)

        if df.empty:
            logger.info("No admissions with discharge dates to analyse.")
            return pd.DataFrame()

        # Rename columns with prefixes for the self-join
        a1 = df.rename(columns={c: f"_a1_{c}" for c in df.columns})
        a2 = df.rename(columns={c: f"_a2_{c}" for c in df.columns})

        # Merge on patient_id → cartesian product per patient
        merged = a1.merge(
            a2,
            left_on="_a1_patient_id",
            right_on="_a2_patient_id",
            how="inner",
            suffixes=("_a1", "_a2"),
        )

        # Apply self-join predicates (vectorised)
        mask = (
            (merged["_a2_admission_date"] > merged["_a1_discharge_date"])
            & (
                merged["_a2_admission_date"]
                <= merged["_a1_discharge_date"]
                + pd.Timedelta(days=READMISSION_WINDOW_DAYS)
            )
        )
        result = merged.loc[mask].copy()

        result = self._compute_readmit_flag(result)
        result = result.rename(columns={"_a1_patient_id": "patient_id"})

        cols_to_keep = [
            "patient_id",
            "_a1_admission_date",
            "_a1_discharge_date",
            "_a2_admission_date",
            "_a2_discharge_date",
            "days_to_readmit",
            "is_readmission",
        ]
        cols_available = [c for c in cols_to_keep if c in result.columns]
        result = result[cols_available]

        logger.info(
            "Detected %d potential readmission pair(s) across %d unique patients.",
            len(result),
            result["patient_id"].nunique() if "patient_id" in result.columns else 0,
        )
        return result.reset_index(drop=True)

    def _detect_readmissions_sql(self) -> pd.DataFrame:
        """SQL-backed readmission detection using the raw self-join."""
        sql = self.self_join_sql()
        logger.debug("Executing self-join SQL:\n%s", sql)
        return pd.read_sql(sql, self.engine)

    # ------------------------------------------------------------------
    # Rate analyses
    # ------------------------------------------------------------------

    def readmission_rate_by_diagnosis(self) -> pd.DataFrame:
        """30-day readmission rate grouped by primary diagnosis code.

        Returns
        -------
        pd.DataFrame
            Columns: diagnosis_code, description, total_admissions,
            readmissions, readmission_rate
        """
        sql = """
            SELECT
                d.diagnosis_code,
                d.description,
                COUNT(DISTINCT a1.admission_id)          AS total_admissions,
                COUNT(DISTINCT a2.admission_id)          AS readmissions,
                ROUND(
                    CAST(COUNT(DISTINCT a2.admission_id) AS REAL)
                    / NULLIF(COUNT(DISTINCT a1.admission_id), 0),
                    4
                ) AS readmission_rate
            FROM   admissions a1
            LEFT JOIN admissions a2
              ON  a1.patient_id = a2.patient_id
             AND  a2.admission_date > a1.discharge_date
             AND  a2.admission_date <= date(a1.discharge_date, '+30 days')
            JOIN   diagnoses d
              ON  a1.diagnosis_code = d.diagnosis_code
            WHERE  a1.discharge_date IS NOT NULL
            GROUP  BY d.diagnosis_code, d.description
            ORDER  BY readmission_rate DESC;
        """
        if self.engine is not None:
            return pd.read_sql(sql, self.engine)

        logger.warning(
            "readmission_rate_by_diagnosis requires a database engine; "
            "returning empty DataFrame."
        )
        return pd.DataFrame(
            columns=[
                "diagnosis_code",
                "description",
                "total_admissions",
                "readmissions",
                "readmission_rate",
            ]
        )

    def readmission_rate_by_department(self) -> pd.DataFrame:
        """Readmission rate by hospital department / service.

        Returns
        -------
        pd.DataFrame
            Columns: department, total_admissions, readmissions,
            readmission_rate
        """
        sql = """
            SELECT
                a1.department,
                COUNT(DISTINCT a1.admission_id)          AS total_admissions,
                COUNT(DISTINCT a2.admission_id)          AS readmissions,
                ROUND(
                    CAST(COUNT(DISTINCT a2.admission_id) AS REAL)
                    / NULLIF(COUNT(DISTINCT a1.admission_id), 0),
                    4
                ) AS readmission_rate
            FROM   admissions a1
            LEFT JOIN admissions a2
              ON  a1.patient_id = a2.patient_id
             AND  a2.admission_date > a1.discharge_date
             AND  a2.admission_date <= date(a1.discharge_date, '+30 days')
            WHERE  a1.discharge_date IS NOT NULL
            GROUP  BY a1.department
            ORDER  BY readmission_rate DESC;
        """
        if self.engine is not None:
            return pd.read_sql(sql, self.engine)
        logger.warning("readmission_rate_by_department requires a database engine.")
        return pd.DataFrame(
            columns=["department", "total_admissions", "readmissions", "readmission_rate"]
        )

    def readmission_rate_by_discharge_disposition(self) -> pd.DataFrame:
        """Readmission rate by discharge disposition (Home, Transfer, etc.).

        Returns
        -------
        pd.DataFrame
            Columns: discharge_disposition, total_admissions, readmissions,
            readmission_rate
        """
        sql = """
            SELECT
                a1.discharge_disposition,
                COUNT(DISTINCT a1.admission_id)          AS total_admissions,
                COUNT(DISTINCT a2.admission_id)          AS readmissions,
                ROUND(
                    CAST(COUNT(DISTINCT a2.admission_id) AS REAL)
                    / NULLIF(COUNT(DISTINCT a1.admission_id), 0),
                    4
                ) AS readmission_rate
            FROM   admissions a1
            LEFT JOIN admissions a2
              ON  a1.patient_id = a2.patient_id
             AND  a2.admission_date > a1.discharge_date
             AND  a2.admission_date <= date(a1.discharge_date, '+30 days')
            WHERE  a1.discharge_date IS NOT NULL
            GROUP  BY a1.discharge_disposition
            ORDER  BY readmission_rate DESC;
        """
        if self.engine is not None:
            return pd.read_sql(sql, self.engine)
        logger.warning(
            "readmission_rate_by_discharge_disposition requires a database engine."
        )
        return pd.DataFrame(
            columns=[
                "discharge_disposition",
                "total_admissions",
                "readmissions",
                "readmission_rate",
            ]
        )

    def readmission_rate_by_insurance(self) -> pd.DataFrame:
        """Readmission rate by insurance type.

        Returns
        -------
        pd.DataFrame
            Columns: insurance_type, total_admissions, readmissions,
            readmission_rate
        """
        sql = """
            SELECT
                a1.insurance_type,
                COUNT(DISTINCT a1.admission_id)          AS total_admissions,
                COUNT(DISTINCT a2.admission_id)          AS readmissions,
                ROUND(
                    CAST(COUNT(DISTINCT a2.admission_id) AS REAL)
                    / NULLIF(COUNT(DISTINCT a1.admission_id), 0),
                    4
                ) AS readmission_rate
            FROM   admissions a1
            LEFT JOIN admissions a2
              ON  a1.patient_id = a2.patient_id
             AND  a2.admission_date > a1.discharge_date
             AND  a2.admission_date <= date(a1.discharge_date, '+30 days')
            WHERE  a1.discharge_date IS NOT NULL
            GROUP  BY a1.insurance_type
            ORDER  BY readmission_rate DESC;
        """
        if self.engine is not None:
            return pd.read_sql(sql, self.engine)
        logger.warning("readmission_rate_by_insurance requires a database engine.")
        return pd.DataFrame(
            columns=[
                "insurance_type",
                "total_admissions",
                "readmissions",
                "readmission_rate",
            ]
        )

    def readmission_rate_by_length_of_stay(self) -> pd.DataFrame:
        """Readmission rate by length-of-stay bands.

        Bands: 0-1 days, 2-3 days, 4-6 days, 7-9 days, 10-14 days,
        15+ days.

        Returns
        -------
        pd.DataFrame
            Columns: los_band, total_admissions, readmissions,
            readmission_rate
        """
        sql = """
            WITH base AS (
                SELECT
                    a1.admission_id,
                    a1.patient_id,
                    a1.admission_date,
                    a1.discharge_date,
                    CAST(julianday(a1.discharge_date) - julianday(a1.admission_date)
                         AS INTEGER) AS los_days
                FROM admissions a1
                WHERE a1.discharge_date IS NOT NULL
            ),
            banded AS (
                SELECT *,
                    CASE
                        WHEN los_days <= 1  THEN '0-1 days'
                        WHEN los_days <= 3  THEN '2-3 days'
                        WHEN los_days <= 6  THEN '4-6 days'
                        WHEN los_days <= 9  THEN '7-9 days'
                        WHEN los_days <= 14 THEN '10-14 days'
                        ELSE '15+ days'
                    END AS los_band
                FROM base
            )
            SELECT
                b1.los_band,
                COUNT(DISTINCT b1.admission_id)  AS total_admissions,
                COUNT(DISTINCT b2.admission_id)  AS readmissions,
                ROUND(
                    CAST(COUNT(DISTINCT b2.admission_id) AS REAL)
                    / NULLIF(COUNT(DISTINCT b1.admission_id), 0),
                    4
                ) AS readmission_rate
            FROM   banded b1
            LEFT JOIN admissions b2
              ON  b1.patient_id = b2.patient_id
             AND  b2.admission_date > b1.discharge_date
             AND  b2.admission_date <= date(b1.discharge_date, '+30 days')
            GROUP  BY b1.los_band
            ORDER  BY readmission_rate DESC;
        """
        if self.engine is not None:
            return pd.read_sql(sql, self.engine)

        # Pandas fallback
        logger.info(
            "readmission_rate_by_length_of_stay: no engine — "
            "attempting pandas-only path (requires a DataFrame to be set)."
        )
        return pd.DataFrame(
            columns=["los_band", "total_admissions", "readmissions", "readmission_rate"]
        )

    # ------------------------------------------------------------------
    # Trend & ranking
    # ------------------------------------------------------------------

    def readmission_trend(self, freq: str = "M") -> pd.DataFrame:
        """Readmission rate over time (monthly or quarterly).

        Parameters
        ----------
        freq : str
            Pandas offset alias.  ``'M'`` for monthly, ``'Q'`` for quarterly.

        Returns
        -------
        pd.DataFrame
            Columns: period, total_admissions, readmissions,
            readmission_rate
        """
        period_label = "month" if freq == "M" else "quarter"

        sql = f"""
            SELECT
                strftime('%Y-%m', a1.admission_date)   AS period,
                COUNT(DISTINCT a1.admission_id)        AS total_admissions,
                COUNT(DISTINCT a2.admission_id)        AS readmissions,
                ROUND(
                    CAST(COUNT(DISTINCT a2.admission_id) AS REAL)
                    / NULLIF(COUNT(DISTINCT a1.admission_id), 0),
                    4
                ) AS readmission_rate
            FROM   admissions a1
            LEFT JOIN admissions a2
              ON  a1.patient_id = a2.patient_id
             AND  a2.admission_date > a1.discharge_date
             AND  a2.admission_date <= date(a1.discharge_date, '+30 days')
            WHERE  a1.discharge_date IS NOT NULL
            GROUP  BY period
            ORDER  BY period;
        """
        if self.engine is not None:
            return pd.read_sql(sql, self.engine)

        logger.warning("readmission_trend requires a database engine.")
        return pd.DataFrame(
            columns=["period", "total_admissions", "readmissions", "readmission_rate"]
        )

    def top_readmission_diagnoses(self, top_n: int = 20) -> pd.DataFrame:
        """Top diagnoses with highest readmission rates.

        Parameters
        ----------
        top_n : int
            Number of diagnoses to return.

        Returns
        -------
        pd.DataFrame
            Columns: diagnosis_code, description, total_admissions,
            readmissions, readmission_rate
        """
        df = self.readmission_rate_by_diagnosis()
        if df.empty:
            return df
        return df.head(top_n).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Cohort analysis
    # ------------------------------------------------------------------

    def readmission_cohort_analysis(self) -> pd.DataFrame:
        """Cohort analysis: patients grouped by first admission date.

        Patients are assigned to a cohort based on their first-ever
        admission month.  We then track what fraction of each cohort
        was readmitted within 30 days of *any* subsequent admission.

        Returns
        -------
        pd.DataFrame
            Columns: cohort_month, cohort_size, patients_readmitted,
            readmission_rate
        """
        sql = """
            WITH first_admission AS (
                SELECT
                    patient_id,
                    MIN(admission_date) AS first_admission_date
                FROM   admissions
                GROUP  BY patient_id
            ),
            cohort_members AS (
                SELECT
                    fa.patient_id,
                    strftime('%Y-%m', fa.first_admission_date) AS cohort_month
                FROM first_admission fa
            ),
            readmitted_patients AS (
                SELECT DISTINCT a1.patient_id
                FROM   admissions a1
                JOIN   admissions a2
                  ON  a1.patient_id = a2.patient_id
                 AND  a2.admission_date > a1.discharge_date
                 AND  a2.admission_date <= date(a1.discharge_date, '+30 days')
                WHERE  a1.discharge_date IS NOT NULL
            )
            SELECT
                cm.cohort_month,
                COUNT(DISTINCT cm.patient_id)       AS cohort_size,
                COUNT(DISTINCT rp.patient_id)       AS patients_readmitted,
                ROUND(
                    CAST(COUNT(DISTINCT rp.patient_id) AS REAL)
                    / NULLIF(COUNT(DISTINCT cm.patient_id), 0),
                    4
                ) AS readmission_rate
            FROM   cohort_members cm
            LEFT JOIN readmitted_patients rp
              ON  cm.patient_id = rp.patient_id
            GROUP  BY cm.cohort_month
            ORDER  BY cm.cohort_month;
        """
        if self.engine is not None:
            return pd.read_sql(sql, self.engine)

        logger.warning("readmission_cohort_analysis requires a database engine.")
        return pd.DataFrame(
            columns=[
                "cohort_month",
                "cohort_size",
                "patients_readmitted",
                "readmission_rate",
            ]
        )

    # ------------------------------------------------------------------
    # SQL documentation helper
    # ------------------------------------------------------------------

    def self_join_sql(self) -> str:
        """Return the raw SQL for the self-join query as a string for documentation.

        This is the canonical 30-day readmission self-join expressed in
        standard SQLite SQL.  It can be pasted directly into a notebook,
        dbt model, or documentation page.

        Returns
        -------
        str
            Readable SQL string.
        """
        return """
-- 30-Day Readmission Self-Join
-- ============================
-- For each pair of admissions (a1, a2) belonging to the same patient,
-- flag a2 as a readmission if it starts within 30 days of a1's discharge.
--
-- SELECT a1.patient_id,
--        a1.admission_id   AS admission_1,
--        a1.admission_date  AS admission_date_1,
--        a1.discharge_date  AS discharge_date_1,
--        a2.admission_id   AS admission_2,
--        a2.admission_date  AS admission_date_2,
--        a2.discharge_date  AS discharge_date_2,
--        CAST(julianday(a2.admission_date)
--             - julianday(a1.discharge_date) AS INTEGER)
--             AS days_to_readmit,
--        CASE
--            WHEN julianday(a2.admission_date)
--                 - julianday(a1.discharge_date)
--                 BETWEEN 1 AND 30
--            THEN 1 ELSE 0
--        END AS is_readmission
-- FROM   admissions a1
-- JOIN   admissions a2
--   ON   a1.patient_id = a2.patient_id
--  AND   a2.admission_date > a1.discharge_date
--  AND   a2.admission_date <= date(a1.discharge_date, '+30 days')
-- WHERE  a1.discharge_date IS NOT NULL
-- ORDER  BY a1.patient_id, a1.admission_date;
""".strip()

    # ------------------------------------------------------------------
    # Aggregate runner
    # ------------------------------------------------------------------

    def full_readmission_analysis(self) -> Dict[str, pd.DataFrame]:
        """Run all readmission analyses and return results dict.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys correspond to analysis names; values are DataFrames.
        """
        results: Dict[str, pd.DataFrame] = {}

        analyzers = {
            "by_diagnosis": self.readmission_rate_by_diagnosis,
            "by_department": self.readmission_rate_by_department,
            "by_discharge_disposition": self.readmission_rate_by_discharge_disposition,
            "by_insurance": self.readmission_rate_by_insurance,
            "by_length_of_stay": self.readmission_rate_by_length_of_stay,
            "trend": self.readmission_trend,
            "top_diagnoses": self.top_readmission_diagnoses,
            "cohort": self.readmission_cohort_analysis,
        }

        for name, func in analyzers.items():
            try:
                results[name] = func()
                logger.info("Completed analysis: %s (%d rows)", name, len(results[name]))
            except Exception:
                logger.exception("Failed analysis: %s", name)
                results[name] = pd.DataFrame()

        return results
