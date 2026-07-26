"""
Provider Performance Analytics Module
=====================================
Provider performance scoring using percentile rankings and composite metrics.
Implements PERCENTILE_CONT equivalent via pandas rank(pct=True) and numpy.percentile.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Default weights for composite score (lower weight = less impact)
DEFAULT_WEIGHTS = {
    "readmission_rate": 0.40,
    "cost": 0.30,
    "los": 0.30,
}

# Minimum patient threshold for reliable provider statistics
MIN_PATIENTS_THRESHOLD = 10


class ProviderPerformanceScorer:
    """Score provider performance using percentile rankings.

    Provides provider-level analytics with composite scoring that combines
    readmission rate, cost, and length-of-stay metrics into a single
    performance score.  Uses PERCENTILE_CONT equivalent via pandas
    ``rank(pct=True)`` for continuous percentile ranking.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Database engine for SQL-backed queries.
    weights : dict, optional
        Custom weights for composite score.  Keys: ``readmission_rate``,
        ``cost``, ``los``.  Defaults to 0.4/0.3/0.3 split.
    min_patients : int
        Minimum number of admissions for a provider to be included in
        rankings (default 10).
    """

    def __init__(
        self,
        engine,
        weights: Optional[Dict[str, float]] = None,
        min_patients: int = MIN_PATIENTS_THRESHOLD,
    ) -> None:
        self.engine = engine
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.min_patients = min_patients

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
        """Compute percentile rank using pandas rank(pct=True).

        PERCENTILE_CONT equivalent: returns a value in [0, 1] representing
        the fraction of values <= the current value.

        Parameters
        ----------
        series : pd.Series
            Numeric values to rank.
        ascending : bool
            If True, lower values get lower ranks.

        Returns
        -------
        pd.Series
            Percentile ranks in [0, 1].
        """
        return series.rank(pct=True, ascending=ascending, na_option="bottom")

    @staticmethod
    def _compute_composite_score(
        ranks: pd.DataFrame,
        weights: Dict[str, float],
    ) -> pd.Series:
        """Weighted sum of percentile ranks.

        Lower composite score = better performance.
        """
        score = pd.Series(0.0, index=ranks.index)
        for metric, weight in weights.items():
            if metric in ranks.columns:
                score += weight * ranks[metric]
        return score

    @staticmethod
    def _assign_category(score: pd.Series) -> pd.Series:
        """Categorize providers into Top 10%, Average, Bottom 10%."""
        q90 = score.quantile(0.10)  # best (lowest) 10%
        q10 = score.quantile(0.90)  # worst (highest) 10%
        conditions = [
            score <= q90,
            score >= q10,
        ]
        choices = ["Top 10%", "Bottom 10%"]
        return pd.Series(
            np.select(conditions, choices, default="Average"),
            index=score.index,
        )

    # ------------------------------------------------------------------
    # Core provider data
    # ------------------------------------------------------------------

    def provider_summary(self) -> pd.DataFrame:
        """Provider-level summary metrics.

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, name, specialty, total_admissions,
            avg_length_of_stay, readmission_rate, avg_cost, avg_charges

        SQL equivalent::

            SELECT
                p.provider_id,
                p.name,
                p.specialty,
                COUNT(a.admission_id) AS total_admissions,
                ROUND(AVG(COALESCE(a.length_of_stay, 0)), 2) AS avg_length_of_stay,
                ROUND(
                    CAST(SUM(CASE
                        WHEN EXISTS (
                            SELECT 1 FROM admissions a2
                            WHERE a2.patient_id = a.patient_id
                              AND a2.admission_date > a.discharge_date
                              AND a2.admission_date <= date(a.discharge_date, '+30 days')
                        ) THEN 1 ELSE 0
                    END) AS REAL)
                    / NULLIF(COUNT(a.admission_id), 0),
                    4
                ) AS readmission_rate,
                ROUND(AVG(COALESCE(a.total_charges, 0)), 2) AS avg_cost,
                ROUND(SUM(COALESCE(a.total_charges, 0)), 2) AS avg_charges
            FROM providers p
            LEFT JOIN admissions a ON p.provider_id = a.provider_id
            GROUP BY p.provider_id, p.name, p.specialty;
        """
        sql = text("""
            SELECT
                p.provider_id,
                p.name,
                p.specialty,
                a.admission_id,
                a.patient_id,
                COALESCE(a.length_of_stay, 0) AS length_of_stay,
                COALESCE(a.total_charges, 0) AS total_cost,
                a.discharge_date
            FROM providers p
            LEFT JOIN admissions a ON p.provider_id = a.provider_id
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "provider_id",
                    "name",
                    "specialty",
                    "total_admissions",
                    "avg_length_of_stay",
                    "readmission_rate",
                    "avg_cost",
                    "avg_charges",
                ]
            )

        # Compute readmission flags per admission (vectorised self-join)
        readmit_df = self._compute_readmission_flags(df)

        # Merge readmission info back
        if not readmit_df.empty:
            df = df.merge(
                readmit_df[["admission_id", "is_readmitted"]],
                on="admission_id",
                how="left",
            )
        else:
            df["is_readmitted"] = 0

        df["is_readmitted"] = df["is_readmitted"].fillna(0).astype(int)

        # Aggregate per provider
        result = (
            df.groupby(["provider_id", "name", "specialty"], dropna=False)
            .agg(
                total_admissions=("admission_id", "count"),
                avg_length_of_stay=("length_of_stay", "mean"),
                readmission_rate=("is_readmitted", "mean"),
                avg_cost=("total_cost", "mean"),
                avg_charges=("total_cost", "sum"),
            )
            .reset_index()
        )

        result["avg_length_of_stay"] = result["avg_length_of_stay"].round(2)
        result["readmission_rate"] = result["readmission_rate"].round(4)
        result["avg_cost"] = result["avg_cost"].round(2)
        result["avg_charges"] = result["avg_charges"].round(2)

        # Filter out providers with too few patients
        before = len(result)
        result = result[result["total_admissions"] >= self.min_patients].copy()
        if len(result) < before:
            logger.info(
                "Filtered out %d providers with < %d admissions.",
                before - len(result),
                self.min_patients,
            )

        logger.info(
            "Computed provider_summary for %d providers (filtered by >= %d admissions).",
            len(result),
            self.min_patients,
        )
        return result.sort_values("readmission_rate", ascending=False).reset_index(
            drop=True
        )

    def _compute_readmission_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorised readmission detection for a provider dataset.

        For each admission, check if the same patient has a subsequent
        admission within 30 days.
        """
        if "discharge_date" not in df.columns:
            return pd.DataFrame()

        valid = df.dropna(subset=["discharge_date"]).copy()
        if valid.empty:
            return pd.DataFrame()

        valid["discharge_date"] = pd.to_datetime(valid["discharge_date"])
        valid["admission_date"] = pd.to_datetime(
            valid.get("admission_date", valid.index)
        )

        # Sort by patient and date for efficient rolling comparison
        valid = valid.sort_values(["patient_id", "admission_date"])

        # Self-join via merge
        a1 = valid[["patient_id", "admission_id", "discharge_date"]].rename(
            columns={"admission_id": "_a1_admission_id", "discharge_date": "_a1_discharge"}
        )
        a2 = valid[["patient_id", "admission_id", "admission_date"]].rename(
            columns={"admission_id": "_a2_admission_id", "admission_date": "_a2_admission"}
        )

        merged = a1.merge(a2, on="patient_id", how="inner")

        mask = (
            (merged["_a2_admission"] > merged["_a1_discharge"])
            & (
                merged["_a2_admission"]
                <= merged["_a1_discharge"] + pd.Timedelta(days=30)
            )
        )
        readmit_pairs = merged.loc[mask, ["_a1_admission_id"]].drop_duplicates()
        readmit_pairs = readmit_pairs.rename(
            columns={"_a1_admission_id": "admission_id"}
        )
        readmit_pairs["is_readmitted"] = 1

        return readmit_pairs

    # ------------------------------------------------------------------
    # Percentile rankings
    # ------------------------------------------------------------------

    def percentile_rank_readmission(self) -> pd.DataFrame:
        """Percentile rank providers by readmission rate (lower is better).

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, name, readmission_rate, percentile_rank

        SQL equivalent::

            SELECT provider_id, name, readmission_rate,
                   PERCENTILE_CONT(readmission_rate)
                       WITHIN GROUP (ORDER BY readmission_rate)
                       OVER () AS percentile_rank
            FROM provider_summary;
        """
        summary = self.provider_summary()
        if summary.empty:
            return pd.DataFrame(
                columns=["provider_id", "name", "readmission_rate", "percentile_rank"]
            )

        result = summary[["provider_id", "name", "readmission_rate"]].copy()
        # Lower readmission rate = better, so ascending=True ranks low values as low
        result["percentile_rank"] = self._percentile_rank(
            result["readmission_rate"], ascending=True
        ).round(4)

        logger.info(
            "Computed percentile_rank_readmission for %d providers.", len(result)
        )
        return result.sort_values("percentile_rank").reset_index(drop=True)

    def percentile_rank_cost(self) -> pd.DataFrame:
        """Percentile rank providers by average cost per episode.

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, name, avg_cost, percentile_rank

        SQL equivalent::

            SELECT provider_id, name, avg_cost,
                   PERCENTILE_CONT(avg_cost)
                       WITHIN GROUP (ORDER BY avg_cost)
                       OVER () AS percentile_rank
            FROM provider_summary;
        """
        summary = self.provider_summary()
        if summary.empty:
            return pd.DataFrame(
                columns=["provider_id", "name", "avg_cost", "percentile_rank"]
            )

        result = summary[["provider_id", "name", "avg_cost"]].copy()
        result["percentile_rank"] = self._percentile_rank(
            result["avg_cost"], ascending=True
        ).round(4)

        logger.info("Computed percentile_rank_cost for %d providers.", len(result))
        return result.sort_values("percentile_rank").reset_index(drop=True)

    def percentile_rank_los(self) -> pd.DataFrame:
        """Percentile rank providers by average length of stay.

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, name, avg_length_of_stay, percentile_rank

        SQL equivalent::

            SELECT provider_id, name, avg_length_of_stay,
                   PERCENTILE_CONT(avg_length_of_stay)
                       WITHIN GROUP (ORDER BY avg_length_of_stay)
                       OVER () AS percentile_rank
            FROM provider_summary;
        """
        summary = self.provider_summary()
        if summary.empty:
            return pd.DataFrame(
                columns=[
                    "provider_id",
                    "name",
                    "avg_length_of_stay",
                    "percentile_rank",
                ]
            )

        result = summary[["provider_id", "name", "avg_length_of_stay"]].copy()
        result["percentile_rank"] = self._percentile_rank(
            result["avg_length_of_stay"], ascending=True
        ).round(4)

        logger.info("Computed percentile_rank_los for %d providers.", len(result))
        return result.sort_values("percentile_rank").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    def composite_performance_score(self) -> pd.DataFrame:
        """Composite score combining readmission rate, cost, and LOS.

        Score = weighted combination of percentile ranks.
        Lower composite = better performance.

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, name, specialty, total_admissions,
            readmission_rate, avg_cost, avg_length_of_stay,
            readmission_rank, cost_rank, los_rank, composite_score

        SQL equivalent (conceptual)::

            WITH ranks AS (
                SELECT *,
                    PERCENTILE_CONT(readmission_rate)
                        WITHIN GROUP (ORDER BY readmission_rate) OVER () AS rr_rank,
                    PERCENTILE_CONT(avg_cost)
                        WITHIN GROUP (ORDER BY avg_cost) OVER () AS cost_rank,
                    PERCENTILE_CONT(avg_length_of_stay)
                        WITHIN GROUP (ORDER BY avg_length_of_stay) OVER () AS los_rank
                FROM provider_summary
            )
            SELECT *,
                :w_rr * rr_rank + :w_cost * cost_rank + :w_los * los_rank
                    AS composite_score
            FROM ranks;
        """
        summary = self.provider_summary()
        if summary.empty:
            return pd.DataFrame(
                columns=[
                    "provider_id",
                    "name",
                    "specialty",
                    "total_admissions",
                    "readmission_rate",
                    "avg_cost",
                    "avg_length_of_stay",
                    "readmission_rank",
                    "cost_rank",
                    "los_rank",
                    "composite_score",
                ]
            )

        result = summary.copy()

        # Compute percentile ranks (lower = better for all metrics)
        result["readmission_rank"] = self._percentile_rank(
            result["readmission_rate"], ascending=True
        )
        result["cost_rank"] = self._percentile_rank(
            result["avg_cost"], ascending=True
        )
        result["los_rank"] = self._percentile_rank(
            result["avg_length_of_stay"], ascending=True
        )

        # Weighted composite score
        ranks_df = result[["readmission_rank", "cost_rank", "los_rank"]].copy()
        result["composite_score"] = self._compute_composite_score(
            ranks_df, self.weights
        ).round(4)

        result["readmission_rank"] = result["readmission_rank"].round(4)
        result["cost_rank"] = result["cost_rank"].round(4)
        result["los_rank"] = result["los_rank"].round(4)

        logger.info(
            "Computed composite_performance_score for %d providers "
            "(weights: %s).",
            len(result),
            self.weights,
        )
        return result.sort_values("composite_score").reset_index(drop=True)

    def provider_ranking(self) -> pd.DataFrame:
        """Overall provider ranking by composite performance score.

        Returns
        -------
        pd.DataFrame
            Columns: rank, provider_id, name, specialty, composite_score,
            category (Top 10%, Average, Bottom 10%)

        SQL equivalent (conceptual)::

            SELECT RANK() OVER (ORDER BY composite_score) AS rank,
                   provider_id, name, specialty, composite_score,
                   CASE
                       WHEN PERCENT_RANK() OVER (ORDER BY composite_score) <= 0.10
                            THEN 'Top 10%'
                       WHEN PERCENT_RANK() OVER (ORDER BY composite_score) >= 0.90
                            THEN 'Bottom 10%'
                       ELSE 'Average'
                   END AS category
            FROM composite_scores
            ORDER BY composite_score;
        """
        composite = self.composite_performance_score()
        if composite.empty:
            return pd.DataFrame(
                columns=[
                    "rank",
                    "provider_id",
                    "name",
                    "specialty",
                    "composite_score",
                    "category",
                ]
            )

        result = composite[
            ["provider_id", "name", "specialty", "composite_score"]
        ].copy()
        result = result.sort_values("composite_score").reset_index(drop=True)
        result["rank"] = result.index + 1
        result["category"] = self._assign_category(result["composite_score"])

        result = result[
            ["rank", "provider_id", "name", "specialty", "composite_score", "category"]
        ]

        logger.info(
            "Computed provider_ranking: Top 10%%=%d, Average=%d, Bottom 10%%=%d.",
            (result["category"] == "Top 10%").sum(),
            (result["category"] == "Average").sum(),
            (result["category"] == "Bottom 10%").sum(),
        )
        return result

    # ------------------------------------------------------------------
    # Specialty and trend analyses
    # ------------------------------------------------------------------

    def performance_by_specialty(self) -> pd.DataFrame:
        """Compare providers within same specialty using percentile ranks.

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, name, specialty, total_admissions,
            readmission_rate, avg_cost, avg_length_of_stay,
            specialty_readmission_rank, specialty_cost_rank, specialty_los_rank,
            specialty_composite_score

        SQL equivalent (conceptual)::

            SELECT *,
                PERCENTILE_CONT(readmission_rate)
                    WITHIN GROUP (ORDER BY readmission_rate)
                    OVER (PARTITION BY specialty) AS specialty_readmission_rank,
                PERCENTILE_CONT(avg_cost)
                    WITHIN GROUP (ORDER BY avg_cost)
                    OVER (PARTITION BY specialty) AS specialty_cost_rank,
                PERCENTILE_CONT(avg_length_of_stay)
                    WITHIN GROUP (ORDER BY avg_length_of_stay)
                    OVER (PARTITION BY specialty) AS specialty_los_rank
            FROM provider_summary;
        """
        summary = self.provider_summary()
        if summary.empty:
            return pd.DataFrame(
                columns=[
                    "provider_id",
                    "name",
                    "specialty",
                    "total_admissions",
                    "readmission_rate",
                    "avg_cost",
                    "avg_length_of_stay",
                    "specialty_readmission_rank",
                    "specialty_cost_rank",
                    "specialty_los_rank",
                    "specialty_composite_score",
                ]
            )

        result = summary.copy()

        # Compute within-specialty percentile ranks
        for metric, col_name in [
            ("readmission_rate", "specialty_readmission_rank"),
            ("avg_cost", "specialty_cost_rank"),
            ("avg_length_of_stay", "specialty_los_rank"),
        ]:
            result[col_name] = (
                result.groupby("specialty")[metric]
                .transform(lambda x: self._percentile_rank(x, ascending=True))
                .round(4)
            )

        # Specialty composite score
        ranks_cols = [
            "specialty_readmission_rank",
            "specialty_cost_rank",
            "specialty_los_rank",
        ]
        result["specialty_composite_score"] = self._compute_composite_score(
            result[ranks_cols], self.weights
        ).round(4)

        logger.info(
            "Computed performance_by_specialty for %d providers across %d specialties.",
            len(result),
            result["specialty"].nunique(),
        )
        return result.sort_values(
            ["specialty", "specialty_composite_score"]
        ).reset_index(drop=True)

    def performance_trend(self, freq: str = "Q") -> pd.DataFrame:
        """Provider performance trends over time (quarterly).

        Parameters
        ----------
        freq : str
            Pandas offset alias.  ``'Q'`` for quarterly, ``'M'`` for monthly.

        Returns
        -------
        pd.DataFrame
            Columns: period, provider_id, name, total_admissions,
            readmission_rate, avg_cost, avg_length_of_stay

        SQL equivalent (conceptual)::

            SELECT
                strftime('%Y-%m', a.admission_date) AS period,
                p.provider_id,
                p.name,
                COUNT(a.admission_id) AS total_admissions,
                ROUND(AVG(CASE WHEN <is_readmitted> THEN 1.0 ELSE 0.0 END), 4)
                    AS readmission_rate,
                ROUND(AVG(COALESCE(a.total_charges, 0)), 2) AS avg_cost,
                ROUND(AVG(COALESCE(a.length_of_stay, 0)), 2) AS avg_length_of_stay
            FROM providers p
            JOIN admissions a ON p.provider_id = a.provider_id
            GROUP BY period, p.provider_id, p.name
            ORDER BY period, p.provider_id;
        """
        sql = text("""
            SELECT
                p.provider_id,
                p.name,
                a.admission_id,
                a.patient_id,
                a.admission_date,
                a.discharge_date,
                COALESCE(a.length_of_stay, 0) AS length_of_stay,
                COALESCE(a.total_charges, 0) AS total_cost
            FROM providers p
            LEFT JOIN admissions a ON p.provider_id = a.provider_id
            WHERE a.admission_date IS NOT NULL
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "period",
                    "provider_id",
                    "name",
                    "total_admissions",
                    "readmission_rate",
                    "avg_cost",
                    "avg_length_of_stay",
                ]
            )

        df["admission_date"] = pd.to_datetime(df["admission_date"])
        df["discharge_date"] = pd.to_datetime(df["discharge_date"])
        df = df.dropna(subset=["admission_date"])

        # Compute readmission flags
        readmit_df = self._compute_readmission_flags(df)
        if not readmit_df.empty:
            df = df.merge(
                readmit_df[["admission_id", "is_readmitted"]],
                on="admission_id",
                how="left",
            )
        else:
            df["is_readmitted"] = 0
        df["is_readmitted"] = df["is_readmitted"].fillna(0).astype(int)

        # Create period column
        df["period"] = df["admission_date"].dt.to_period(freq)

        result = (
            df.groupby(["period", "provider_id", "name"], dropna=False)
            .agg(
                total_admissions=("admission_id", "count"),
                readmission_rate=("is_readmitted", "mean"),
                avg_cost=("total_cost", "mean"),
                avg_length_of_stay=("length_of_stay", "mean"),
            )
            .reset_index()
        )

        result["period"] = result["period"].astype(str)
        result["readmission_rate"] = result["readmission_rate"].round(4)
        result["avg_cost"] = result["avg_cost"].round(2)
        result["avg_length_of_stay"] = result["avg_length_of_stay"].round(2)

        logger.info(
            "Computed performance_trend (%s) for %d provider-periods.",
            freq,
            len(result),
        )
        return result.sort_values(["period", "provider_id"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def outlier_providers(
        self, metric: str = "readmission_rate", threshold: float = 90
    ) -> pd.DataFrame:
        """Identify providers performing worse than threshold percentile.

        Parameters
        ----------
        metric : str
            Column to evaluate: ``readmission_rate``, ``avg_cost``, or
            ``avg_length_of_stay``.
        threshold : float
            Percentile above which a provider is flagged (default 90).

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, name, specialty, <metric>,
            percentile_value, is_outlier

        SQL equivalent (conceptual)::

            WITH ranked AS (
                SELECT *,
                    PERCENTILE_CONT(<metric>)
                        WITHIN GROUP (ORDER BY <metric>) OVER () AS pctl
                FROM provider_summary
            )
            SELECT *, pctl AS percentile_value,
                   CASE WHEN pctl >= :threshold/100 THEN 1 ELSE 0 END AS is_outlier
            FROM ranked;
        """
        valid_metrics = {"readmission_rate", "avg_cost", "avg_length_of_stay"}
        if metric not in valid_metrics:
            raise ValueError(
                f"metric must be one of {valid_metrics}, got '{metric}'"
            )

        summary = self.provider_summary()
        if summary.empty:
            return pd.DataFrame(
                columns=[
                    "provider_id",
                    "name",
                    "specialty",
                    metric,
                    "percentile_value",
                    "is_outlier",
                ]
            )

        result = summary[["provider_id", "name", "specialty", metric]].copy()
        result["percentile_value"] = (
            self._percentile_rank(result[metric], ascending=True) * 100
        ).round(2)
        threshold_value = float(np.percentile(result[metric].dropna(), threshold))
        result["is_outlier"] = (result[metric] > threshold_value).astype(int)

        outliers = result[result["is_outlier"] == 1].copy()

        logger.info(
            "Identified %d outlier providers for '%s' above P%.0f (%.2f).",
            len(outliers),
            metric,
            threshold,
            threshold_value,
        )
        return outliers.sort_values(metric, ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # SQL documentation
    # ------------------------------------------------------------------

    def percentile_sql(self, column: str, order: str = "ASC") -> str:
        """Return PERCENTILE_CONT SQL for documentation.

        Parameters
        ----------
        column : str
            Column name to compute percentile for.
        order : str
            ``'ASC'`` or ``'DESC'`` for sort direction.

        Returns
        -------
        str
            SQL string using PERCENTILE_CONT window function.

        SQL equivalent::

            SELECT
                provider_id,
                name,
                <column>,
                PERCENTILE_CONT(<column>)
                    WITHIN GROUP (ORDER BY <column> <order>)
                    OVER () AS percentile_rank
            FROM provider_summary;
        """
        order_clause = order.upper()
        return (
            f"SELECT\n"
            f"    provider_id,\n"
            f"    name,\n"
            f"    {column},\n"
            f"    PERCENTILE_CONT({column})\n"
            f"        WITHIN GROUP (ORDER BY {column} {order_clause})\n"
            f"        OVER () AS percentile_rank\n"
            f"FROM provider_summary;"
        )

    # ------------------------------------------------------------------
    # Aggregate runner
    # ------------------------------------------------------------------

    def full_provider_analysis(self) -> Dict[str, pd.DataFrame]:
        """Run all provider analyses and return results dict.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys correspond to analysis names; values are DataFrames.
        """
        results: Dict[str, pd.DataFrame] = {}

        analyses = {
            "provider_summary": self.provider_summary,
            "percentile_rank_readmission": self.percentile_rank_readmission,
            "percentile_rank_cost": self.percentile_rank_cost,
            "percentile_rank_los": self.percentile_rank_los,
            "composite_performance_score": self.composite_performance_score,
            "provider_ranking": self.provider_ranking,
            "performance_by_specialty": self.performance_by_specialty,
            "performance_trend_quarterly": lambda: self.performance_trend(freq="Q"),
            "performance_trend_monthly": lambda: self.performance_trend(freq="M"),
            "outlier_providers_readmission": lambda: self.outlier_providers(
                metric="readmission_rate"
            ),
            "outlier_providers_cost": lambda: self.outlier_providers(metric="avg_cost"),
            "outlier_providers_los": lambda: self.outlier_providers(
                metric="avg_length_of_stay"
            ),
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
