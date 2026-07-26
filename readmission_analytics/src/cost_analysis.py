"""
Cost Analysis Module
====================
Cost per episode breakdown analysis for patient readmission analytics.
Implements cost calculations by diagnosis, department, insurance, and
temporal trends using SQLAlchemy queries and pandas transformations.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)


class CostAnalyzer:
    """Analyze cost per episode and cost breakdown.

    Provides methods for decomposing hospital costs across multiple
    dimensions: diagnosis, department, insurance type, and time.  All
    cost columns are sourced from the ``admissions`` table as
    ``Numeric(12, 2)`` columns.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Database engine for SQL-backed queries.
    """

    def __init__(self, engine) -> None:
        self.engine = engine

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_cost_col(series: pd.Series) -> pd.Series:
        """Coerce to float and clip negative charges to zero."""
        return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)

    @staticmethod
    def _compute_cost_per_day(row: pd.Series) -> float:
        """Avoid division by zero for zero-length stays."""
        los = row.get("length_of_stay", 0)
        if pd.isna(los) or los <= 0:
            return 0.0
        return float(row["total_cost"]) / float(los)

    # ------------------------------------------------------------------
    # Core analyses
    # ------------------------------------------------------------------

    def cost_per_episode(self) -> pd.DataFrame:
        """Total cost per admission episode.

        Returns
        -------
        pd.DataFrame
            Columns: admission_id, patient_id, total_cost,
            length_of_stay, cost_per_day

        SQL equivalent::

            SELECT
                admission_id,
                patient_id,
                COALESCE(total_charges, 0) AS total_cost,
                COALESCE(length_of_stay, 1) AS length_of_stay,
                CASE WHEN COALESCE(length_of_stay, 0) > 0
                     THEN COALESCE(total_charges, 0)
                          / CAST(length_of_stay AS REAL)
                     ELSE 0
                END AS cost_per_day
            FROM admissions;
        """
        sql = text("""
            SELECT
                admission_id,
                patient_id,
                COALESCE(total_charges, 0) AS total_cost,
                COALESCE(length_of_stay, 0) AS length_of_stay
            FROM admissions
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "admission_id",
                    "patient_id",
                    "total_cost",
                    "length_of_stay",
                    "cost_per_day",
                ]
            )

        df["total_cost"] = self._safe_cost_col(df["total_cost"])
        df["length_of_stay"] = pd.to_numeric(
            df["length_of_stay"], errors="coerce"
        ).fillna(0).clip(lower=0)

        df["cost_per_day"] = df.apply(self._compute_cost_per_day, axis=1)

        logger.info(
            "Computed cost_per_episode for %d admissions.", len(df)
        )
        return df

    def cost_by_diagnosis(self) -> pd.DataFrame:
        """Average cost per episode grouped by primary diagnosis.

        Returns
        -------
        pd.DataFrame
            Columns: diagnosis_code, description, avg_cost,
            median_cost, total_cost, episode_count
        """
        sql = text("""
            SELECT
                a.primary_diagnosis_code AS diagnosis_code,
                d.description,
                COALESCE(a.total_charges, 0) AS total_cost
            FROM admissions a
            LEFT JOIN diagnoses d
                ON a.primary_diagnosis_code = d.icd_code
            WHERE a.total_charges IS NOT NULL
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "diagnosis_code",
                    "description",
                    "avg_cost",
                    "median_cost",
                    "total_cost",
                    "episode_count",
                ]
            )

        df["total_cost"] = self._safe_cost_col(df["total_cost"])

        result = (
            df.groupby(["diagnosis_code", "description"], dropna=False)
            .agg(
                avg_cost=("total_cost", "mean"),
                median_cost=("total_cost", "median"),
                total_cost=("total_cost", "sum"),
                episode_count=("total_cost", "count"),
            )
            .reset_index()
            .sort_values("total_cost", ascending=False)
        )

        logger.info(
            "Computed cost_by_diagnosis for %d diagnosis groups.",
            len(result),
        )
        return result

    def cost_by_department(self) -> pd.DataFrame:
        """Cost analysis by hospital department/service.

        Returns
        -------
        pd.DataFrame
            Columns: department, avg_cost, median_cost, total_cost,
            episode_count
        """
        sql = text("""
            SELECT
                pr.department,
                COALESCE(a.total_charges, 0) AS total_cost
            FROM admissions a
            JOIN providers pr
                ON a.provider_id = pr.provider_id
            WHERE a.total_charges IS NOT NULL
              AND pr.department IS NOT NULL
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "department",
                    "avg_cost",
                    "median_cost",
                    "total_cost",
                    "episode_count",
                ]
            )

        df["total_cost"] = self._safe_cost_col(df["total_cost"])

        result = (
            df.groupby("department", dropna=False)
            .agg(
                avg_cost=("total_cost", "mean"),
                median_cost=("total_cost", "median"),
                total_cost=("total_cost", "sum"),
                episode_count=("total_cost", "count"),
            )
            .reset_index()
            .sort_values("total_cost", ascending=False)
        )

        logger.info(
            "Computed cost_by_department for %d departments.", len(result)
        )
        return result

    def cost_breakdown_components(self) -> pd.DataFrame:
        """Break down costs into components: medication, lab, procedure, other.

        ``other`` is derived as:
            total_charges - medication_charges - lab_charges - procedure_charges

        Returns
        -------
        pd.DataFrame
            Columns: component, total_cost, avg_cost, pct_of_total
        """
        sql = text("""
            SELECT
                COALESCE(total_charges, 0)          AS total_cost,
                COALESCE(medication_charges, 0)     AS medication_cost,
                COALESCE(lab_charges, 0)            AS lab_cost,
                COALESCE(procedure_charges, 0)      AS procedure_cost
            FROM admissions
            WHERE total_charges IS NOT NULL
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        if df.empty:
            return pd.DataFrame(
                columns=["component", "total_cost", "avg_cost", "pct_of_total"]
            )

        for col in ("total_cost", "medication_cost", "lab_cost", "procedure_cost"):
            df[col] = self._safe_cost_col(df[col])

        df["other_cost"] = (
            df["total_cost"] - df["medication_cost"] - df["lab_cost"] - df["procedure_cost"]
        ).clip(lower=0.0)

        totals = pd.DataFrame(
            {
                "component": ["medication", "lab", "procedure", "other"],
                "total_cost": [
                    df["medication_cost"].sum(),
                    df["lab_cost"].sum(),
                    df["procedure_cost"].sum(),
                    df["other_cost"].sum(),
                ],
            }
        )

        grand_total = totals["total_cost"].sum()
        totals["avg_cost"] = totals["total_cost"] / len(df)
        totals["pct_of_total"] = (
            (totals["total_cost"] / grand_total * 100).round(2) if grand_total > 0 else 0.0
        )

        logger.info(
            "Computed cost_breakdown_components (grand total=%.2f).",
            grand_total,
        )
        return totals

    def cost_trend(self, freq: str = "M") -> pd.DataFrame:
        """Cost trends over time.

        Parameters
        ----------
        freq : str
            Pandas offset alias.  ``'M'`` for monthly, ``'Q'`` for quarterly.

        Returns
        -------
        pd.DataFrame
            Columns: period, total_cost, avg_cost, episode_count
        """
        sql = text("""
            SELECT
                admission_date,
                COALESCE(total_charges, 0) AS total_cost
            FROM admissions
            WHERE total_charges IS NOT NULL
              AND admission_date IS NOT NULL
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        if df.empty:
            return pd.DataFrame(
                columns=["period", "total_cost", "avg_cost", "episode_count"]
            )

        df["total_cost"] = self._safe_cost_col(df["total_cost"])
        df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")
        df = df.dropna(subset=["admission_date"])

        df["period"] = df["admission_date"].dt.to_period(freq)

        result = (
            df.groupby("period")
            .agg(
                total_cost=("total_cost", "sum"),
                avg_cost=("total_cost", "mean"),
                episode_count=("total_cost", "count"),
            )
            .reset_index()
        )
        result["period"] = result["period"].astype(str)

        logger.info("Computed cost_trend (%s) for %d periods.", freq, len(result))
        return result

    def cost_vs_readmission(self) -> pd.DataFrame:
        """Compare costs between readmitted vs non-readmitted patients.

        Returns
        -------
        pd.DataFrame
            Columns: is_readmitted, avg_cost, median_cost, total_cost,
            episode_count
        """
        sql = text("""
            SELECT
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM admissions a2
                        WHERE a2.patient_id = a.patient_id
                          AND a2.admission_date > a.discharge_date
                          AND a2.admission_date <= date(a.discharge_date, '+30 days')
                    ) THEN 1
                    ELSE 0
                END AS is_readmitted,
                COALESCE(a.total_charges, 0) AS total_cost
            FROM admissions a
            WHERE a.total_charges IS NOT NULL
              AND a.discharge_date IS NOT NULL
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "is_readmitted",
                    "avg_cost",
                    "median_cost",
                    "total_cost",
                    "episode_count",
                ]
            )

        df["total_cost"] = self._safe_cost_col(df["total_cost"])
        df["is_readmitted"] = df["is_readmitted"].astype(int)

        result = (
            df.groupby("is_readmitted")
            .agg(
                avg_cost=("total_cost", "mean"),
                median_cost=("total_cost", "median"),
                total_cost=("total_cost", "sum"),
                episode_count=("total_cost", "count"),
            )
            .reset_index()
        )

        logger.info(
            "Computed cost_vs_readmission: readmitted=%d, not_readmitted=%d.",
            int(result.loc[result["is_readmitted"] == 1, "episode_count"].sum()),
            int(result.loc[result["is_readmitted"] == 0, "episode_count"].sum()),
        )
        return result

    def cost_by_insurance(self) -> pd.DataFrame:
        """Cost analysis by insurance type.

        Returns
        -------
        pd.DataFrame
            Columns: insurance_type, avg_cost, median_cost, total_cost,
            episode_count
        """
        sql = text("""
            SELECT
                COALESCE(p.insurance_type, 'Unknown') AS insurance_type,
                COALESCE(a.total_charges, 0)          AS total_cost
            FROM admissions a
            JOIN patients p
                ON a.patient_id = p.patient_id
            WHERE a.total_charges IS NOT NULL
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "insurance_type",
                    "avg_cost",
                    "median_cost",
                    "total_cost",
                    "episode_count",
                ]
            )

        df["total_cost"] = self._safe_cost_col(df["total_cost"])

        result = (
            df.groupby("insurance_type", dropna=False)
            .agg(
                avg_cost=("total_cost", "mean"),
                median_cost=("total_cost", "median"),
                total_cost=("total_cost", "sum"),
                episode_count=("total_cost", "count"),
            )
            .reset_index()
            .sort_values("total_cost", ascending=False)
        )

        logger.info(
            "Computed cost_by_insurance for %d insurance types.", len(result)
        )
        return result

    def outlier_cost_episodes(
        self, threshold_percentile: float = 95
    ) -> pd.DataFrame:
        """Identify cost outlier episodes above threshold percentile.

        Parameters
        ----------
        threshold_percentile : float
            Percentile above which an episode is flagged as an outlier
            (default 95).

        Returns
        -------
        pd.DataFrame
            Columns: admission_id, patient_id, total_cost,
            length_of_stay, cost_percentile
        """
        base = self.cost_per_episode()
        if base.empty:
            return base

        threshold = float(np.percentile(base["total_cost"], threshold_percentile))
        outliers = base[base["total_cost"] > threshold].copy()

        if not outliers.empty:
            outliers["cost_percentile"] = (
                outliers["total_cost"]
                .rank(pct=True)
                .multiply(100)
                .round(2)
            )

        logger.info(
            "Identified %d outlier episodes above P%.0f (%.2f).",
            len(outliers),
            threshold_percentile,
            threshold,
        )
        return outliers

    def cost_per_diagnosis_group(self) -> pd.DataFrame:
        """Cost per episode grouped by ICD chapter/category.

        Returns
        -------
        pd.DataFrame
            Columns: icd_chapter, avg_cost, median_cost, total_cost,
            episode_count
        """
        sql = text("""
            SELECT
                a.primary_diagnosis_code AS diagnosis_code,
                COALESCE(a.total_charges, 0) AS total_cost
            FROM admissions a
            WHERE a.total_charges IS NOT NULL
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "icd_chapter",
                    "avg_cost",
                    "median_cost",
                    "total_cost",
                    "episode_count",
                ]
            )

        df["total_cost"] = self._safe_cost_col(df["total_cost"])

        from .icd_grouping import ICDGrouper

        grouper = ICDGrouper()
        df["icd_chapter"] = df["diagnosis_code"].map(grouper.get_chapter)

        result = (
            df.groupby("icd_chapter", dropna=False)
            .agg(
                avg_cost=("total_cost", "mean"),
                median_cost=("total_cost", "median"),
                total_cost=("total_cost", "sum"),
                episode_count=("total_cost", "count"),
            )
            .reset_index()
            .sort_values("total_cost", ascending=False)
        )

        logger.info(
            "Computed cost_per_diagnosis_group for %d chapters.", len(result)
        )
        return result

    def percentile_cost_distribution(self) -> pd.DataFrame:
        """Cost distribution with percentiles (P10, P25, P50, P75, P90, P95, P99).

        Uses ``numpy.percentile`` as the PERCENTILE_CONT equivalent.

        Returns
        -------
        pd.DataFrame
            Columns: metric, value
        """
        base = self.cost_per_episode()
        if base.empty:
            return pd.DataFrame(columns=["metric", "value"])

        costs = base["total_cost"].dropna()
        costs = costs[costs >= 0]

        percentiles = [10, 25, 50, 75, 90, 95, 99]
        values = np.percentile(costs, percentiles)

        result = pd.DataFrame(
            {
                "metric": [f"P{p}" for p in percentiles] + [
                    "mean", "std", "min", "max", "count"
                ],
                "values": list(values)
                    + [
                        float(costs.mean()),
                        float(costs.std()),
                        float(costs.min()),
                        float(costs.max()),
                        float(len(costs)),
                    ],
            }
        )

        logger.info(
            "Computed percentile_cost_distribution for %d episodes.", len(costs)
        )
        return result

    # ------------------------------------------------------------------
    # Aggregate runner
    # ------------------------------------------------------------------

    def full_cost_analysis(self) -> Dict[str, pd.DataFrame]:
        """Run all cost analyses and return results dict.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys correspond to analysis names; values are DataFrames.
        """
        results: Dict[str, pd.DataFrame] = {}

        analyses = {
            "cost_per_episode": self.cost_per_episode,
            "cost_by_diagnosis": self.cost_by_diagnosis,
            "cost_by_department": self.cost_by_department,
            "cost_breakdown_components": self.cost_breakdown_components,
            "cost_trend_monthly": lambda: self.cost_trend(freq="M"),
            "cost_trend_quarterly": lambda: self.cost_trend(freq="Q"),
            "cost_vs_readmission": self.cost_vs_readmission,
            "cost_by_insurance": self.cost_by_insurance,
            "cost_per_diagnosis_group": self.cost_per_diagnosis_group,
            "percentile_cost_distribution": self.percentile_cost_distribution,
        }

        for name, func in analyses.items():
            try:
                results[name] = func()
                logger.info(
                    "Completed analysis: %s (%d rows)", name, len(results[name])
                )
            except Exception:
                logger.exception("Failed analysis: %s", name)
                results[name] = pd.DataFrame()

        return results
