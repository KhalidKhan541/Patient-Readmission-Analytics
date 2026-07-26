"""Data quality checks for healthcare data before analysis."""

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd


class HealthcareDataQuality:
    """Data quality checks for healthcare data before analysis."""

    def __init__(self, config: dict):
        self.config = config
        self.results: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def _log(self, check_name: str, passed: bool, details: str = "") -> None:
        """Log and record a check result."""
        status = "PASSED" if passed else "FAILED"
        message = f"[{status}] {check_name}: {details}" if details else f"[{status}] {check_name}"
        if passed:
            self.logger.debug(message)
        else:
            self.logger.warning(message)
        self.results.append({"check": check_name, "passed": passed, "details": details})

    def check_not_null(self, df: pd.DataFrame, columns: List[str]) -> bool:
        """Check columns have no nulls.

        Args:
            df: DataFrame to validate.
            columns: Column names to check for null values.

        Returns:
            True if all specified columns have no nulls, False otherwise.
        """
        if df.empty:
            self._log("check_not_null", True, "DataFrame is empty; skipping.")
            return True

        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            self._log("check_not_null", False, f"Columns not found: {missing_cols}")
            return False

        null_counts = {c: int(df[c].isnull().sum()) for c in columns}
        failed = {c: n for c, n in null_counts.items() if n > 0}
        if failed:
            details = ", ".join(f"{c}: {n} nulls" for c, n in failed.items())
            self._log("check_not_null", False, details)
            return False

        self._log("check_not_null", True)
        return True

    def check_unique(self, df: pd.DataFrame, columns: List[str]) -> bool:
        """Check columns have unique values.

        Args:
            df: DataFrame to validate.
            columns: Column names expected to contain unique values.

        Returns:
            True if all specified columns have unique values, False otherwise.
        """
        if df.empty:
            self._log("check_unique", True, "DataFrame is empty; skipping.")
            return True

        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            self._log("check_unique", False, f"Columns not found: {missing_cols}")
            return False

        failed = {}
        for c in columns:
            dup_count = int(df[c].duplicated().sum())
            if dup_count > 0:
                failed[c] = dup_count

        if failed:
            details = ", ".join(f"{c}: {n} duplicates" for c, n in failed.items())
            self._log("check_unique", False, details)
            return False

        self._log("check_unique", True)
        return True

    def check_date_order(self, df: pd.DataFrame, date1: str, date2: str) -> bool:
        """Check date1 <= date2 for all rows (e.g., admission <= discharge).

        Args:
            df: DataFrame to validate.
            date1: Name of the earlier-date column.
            date2: Name of the later-date column.

        Returns:
            True if date1 <= date2 for every row, False otherwise.
        """
        if df.empty:
            self._log("check_date_order", True, "DataFrame is empty; skipping.")
            return True

        missing = [c for c in (date1, date2) if c not in df.columns]
        if missing:
            self._log("check_date_order", False, f"Columns not found: {missing}")
            return False

        d1 = pd.to_datetime(df[date1], errors="coerce")
        d2 = pd.to_datetime(df[date2], errors="coerce")

        nat_mask = d1.isnull() | d2.isnull()
        nat_count = int(nat_mask.sum())
        if nat_count > 0:
            self._log("check_date_order", False, f"{nat_count} rows have unparseable dates.")
            return False

        violations = int((d1 > d2).sum())
        if violations > 0:
            self._log("check_date_order", False, f"{violations} rows where {date1} > {date2}.")
            return False

        self._log("check_date_order", True)
        return True

    def check_valid_icd(self, df: pd.DataFrame, column: str) -> bool:
        """Check ICD codes match expected format (letter + digits).

        Accepts ICD-9 (e.g., V29.0, E950.1, 250.00) and ICD-10
        (e.g., A01.0, I10, Z23) patterns.

        Args:
            df: DataFrame to validate.
            column: Column containing ICD codes.

        Returns:
            True if all non-null codes match a valid ICD pattern, False otherwise.
        """
        if df.empty:
            self._log("check_valid_icd", True, "DataFrame is empty; skipping.")
            return True

        if column not in df.columns:
            self._log("check_valid_icd", False, f"Column '{column}' not found.")
            return False

        # ICD-9: optional letter prefix (V/E) + 2-3 digits + optional decimal + 0-2 digits
        # ICD-10: letter + 2 alphanumeric chars + optional decimal + 1-4 alphanumeric chars
        icd_pattern = re.compile(
            r"^[A-Za-z]?\d{2,3}(\.\d{1,2})?$"  # ICD-9 basic
            r"|^[A-Za-z]\d{2}(\.\d{1,4})?$",     # ICD-10 basic
            re.IGNORECASE,
        )

        non_null = df[column].dropna()
        if non_null.empty:
            self._log("check_valid_icd", True, "All values are null; skipping.")
            return True

        invalid_mask = ~non_null.astype(str).str.strip().apply(lambda v: bool(icd_pattern.match(v)))
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            examples = non_null[invalid_mask].head(5).tolist()
            self._log("check_valid_icd", False, f"{invalid_count} invalid ICD codes, e.g. {examples}")
            return False

        self._log("check_valid_icd", True)
        return True

    def check_valid_npi(self, df: pd.DataFrame, column: str) -> bool:
        """Check NPI numbers are exactly 10 digits.

        Args:
            df: DataFrame to validate.
            column: Column containing NPI numbers.

        Returns:
            True if all non-null NPIs are 10-digit strings, False otherwise.
        """
        if df.empty:
            self._log("check_valid_npi", True, "DataFrame is empty; skipping.")
            return True

        if column not in df.columns:
            self._log("check_valid_npi", False, f"Column '{column}' not found.")
            return False

        non_null = df[column].dropna()
        if non_null.empty:
            self._log("check_valid_npi", True, "All values are null; skipping.")
            return True

        npi_str = non_null.astype(str).str.strip()
        invalid = ~npi_str.str.match(r"^\d{10}$")
        invalid_count = int(invalid.sum())
        if invalid_count > 0:
            examples = non_null[invalid].head(5).tolist()
            self._log("check_valid_npi", False, f"{invalid_count} invalid NPIs, e.g. {examples}")
            return False

        self._log("check_valid_npi", True)
        return True

    def check_valid_mrn(self, df: pd.DataFrame, column: str) -> bool:
        """Check MRN format.

        MRN is expected to be alphanumeric, typically 6-10 characters.

        Args:
            df: DataFrame to validate.
            column: Column containing MRN values.

        Returns:
            True if all non-null MRNs match expected format, False otherwise.
        """
        if df.empty:
            self._log("check_valid_mrn", True, "DataFrame is empty; skipping.")
            return True

        if column not in df.columns:
            self._log("check_valid_mrn", False, f"Column '{column}' not found.")
            return False

        non_null = df[column].dropna()
        if non_null.empty:
            self._log("check_valid_mrn", True, "All values are null; skipping.")
            return True

        mrn_str = non_null.astype(str).str.strip()
        invalid = ~mrn_str.str.match(r"^[A-Za-z0-9]{6,10}$")
        invalid_count = int(invalid.sum())
        if invalid_count > 0:
            examples = non_null[invalid].head(5).tolist()
            self._log("check_valid_mrn", False, f"{invalid_count} invalid MRNs, e.g. {examples}")
            return False

        self._log("check_valid_mrn", True)
        return True

    def check_range(
        self,
        df: pd.DataFrame,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> bool:
        """Check column values are within range.

        Args:
            df: DataFrame to validate.
            column: Column to check.
            min_val: Minimum allowed value (inclusive). None means no lower bound.
            max_val: Maximum allowed value (inclusive). None means no upper bound.

        Returns:
            True if all non-null values are within range, False otherwise.
        """
        if df.empty:
            self._log("check_range", True, "DataFrame is empty; skipping.")
            return True

        if column not in df.columns:
            self._log("check_range", False, f"Column '{column}' not found.")
            return False

        series = pd.to_numeric(df[column], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            self._log("check_range", True, "All values are null; skipping.")
            return True

        violations: List[str] = []
        if min_val is not None:
            below = int((non_null < min_val).sum())
            if below > 0:
                violations.append(f"{below} values < {min_val}")
        if max_val is not None:
            above = int((non_null > max_val).sum())
            if above > 0:
                violations.append(f"{above} values > {max_val}")

        if violations:
            self._log("check_range", False, "; ".join(violations))
            return False

        self._log("check_range", True)
        return True

    def check_values_in_set(self, df: pd.DataFrame, column: str, valid_values: List) -> bool:
        """Check column values are in valid set.

        Args:
            df: DataFrame to validate.
            column: Column to check.
            valid_values: Allowed values.

        Returns:
            True if all non-null values are in the valid set, False otherwise.
        """
        if df.empty:
            self._log("check_values_in_set", True, "DataFrame is empty; skipping.")
            return True

        if column not in df.columns:
            self._log("check_values_in_set", False, f"Column '{column}' not found.")
            return False

        non_null = df[column].dropna()
        if non_null.empty:
            self._log("check_values_in_set", True, "All values are null; skipping.")
            return True

        valid_set = set(valid_values)
        invalid_mask = ~non_null.isin(valid_set)
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            unexpected = non_null[invalid_mask].unique()[:10].tolist()
            self._log("check_values_in_set", False, f"{invalid_count} invalid values, e.g. {unexpected}")
            return False

        self._log("check_values_in_set", True)
        return True

    def check_referential_integrity(
        self,
        df: pd.DataFrame,
        column: str,
        reference_df: pd.DataFrame,
        ref_column: str,
    ) -> bool:
        """Check foreign key references exist in the reference table.

        Args:
            df: Child DataFrame.
            column: Foreign key column in child.
            reference_df: Parent (reference) DataFrame.
            ref_column: Primary key column in reference.

        Returns:
            True if all non-null FK values exist in the reference, False otherwise.
        """
        if df.empty:
            self._log("check_referential_integrity", True, "Child DataFrame is empty; skipping.")
            return True

        for label, frame, col in [
            ("child", df, column),
            ("reference", reference_df, ref_column),
        ]:
            if col not in frame.columns:
                self._log("check_referential_integrity", False, f"{label} column '{col}' not found.")
                return False

        non_null = df[column].dropna()
        if non_null.empty:
            self._log("check_referential_integrity", True, "No FK values to check; skipping.")
            return True

        ref_keys = set(reference_df[ref_column].dropna())
        orphans = non_null[~non_null.isin(ref_keys)]
        orphan_count = int(len(orphans))
        if orphan_count > 0:
            examples = orphans.unique()[:10].tolist()
            self._log("check_referential_integrity", False, f"{orphan_count} orphan references, e.g. {examples}")
            return False

        self._log("check_referential_integrity", True)
        return True

    def check_temporal_consistency(self, df: pd.DataFrame, start_col: str, end_col: str) -> bool:
        """Check start < end for all rows.

        Unlike check_date_order (which allows equality), this enforces strict
        ordering — useful for time-windowed analyses.

        Args:
            df: DataFrame to validate.
            start_col: Column containing start timestamps.
            end_col: Column containing end timestamps.

        Returns:
            True if start < end for every row, False otherwise.
        """
        if df.empty:
            self._log("check_temporal_consistency", True, "DataFrame is empty; skipping.")
            return True

        missing = [c for c in (start_col, end_col) if c not in df.columns]
        if missing:
            self._log("check_temporal_consistency", False, f"Columns not found: {missing}")
            return False

        s = pd.to_datetime(df[start_col], errors="coerce")
        e = pd.to_datetime(df[end_col], errors="coerce")

        nat_count = int((s.isnull() | e.isnull()).sum())
        if nat_count > 0:
            self._log("check_temporal_consistency", False, f"{nat_count} rows have unparseable dates.")
            return False

        violations = int((s >= e).sum())
        if violations > 0:
            self._log("check_temporal_consistency", False, f"{violations} rows where {start_col} >= {end_col}.")
            return False

        self._log("check_temporal_consistency", True)
        return True

    # ── Table-level check suites ──────────────────────────────────────

    def run_patient_checks(self, patients: pd.DataFrame) -> Dict[str, bool]:
        """Run all patient table quality checks.

        Validates patient_id uniqueness, not-null required fields, MRN format,
        and age range.

        Args:
            patients: Patient dimension DataFrame.

        Returns:
            Mapping of check name to pass/fail.
        """
        self.logger.info("Running patient table checks (%d rows).", len(patients))
        results: Dict[str, bool] = {}

        results["patient_id_unique"] = self.check_unique(patients, ["patient_id"])
        results["patient_id_not_null"] = self.check_not_null(patients, ["patient_id"])
        results["required_fields_not_null"] = self.check_not_null(
            patients,
            [c for c in ("first_name", "last_name", "date_of_birth", "gender") if c in patients.columns],
        )
        if "mrn" in patients.columns:
            results["mrn_valid"] = self.check_valid_mrn(patients, "mrn")
        if "age" in patients.columns:
            results["age_range"] = self.check_range(patients, "age", min_val=0, max_val=120)

        return results

    def run_admission_checks(self, admissions: pd.DataFrame) -> Dict[str, bool]:
        """Run all admission table quality checks.

        Validates admission/discharge date ordering, ICD codes, NPI, and
        required fields.

        Args:
            admissions: Admission fact DataFrame.

        Returns:
            Mapping of check name to pass/fail.
        """
        self.logger.info("Running admission table checks (%d rows).", len(admissions))
        results: Dict[str, bool] = {}

        results["admission_id_unique"] = self.check_unique(admissions, ["admission_id"])
        results["admission_id_not_null"] = self.check_not_null(admissions, ["admission_id"])
        results["required_fields_not_null"] = self.check_not_null(
            admissions,
            [c for c in ("patient_id", "admission_date", "discharge_date") if c in admissions.columns],
        )

        if "admission_date" in admissions.columns and "discharge_date" in admissions.columns:
            results["admission_before_discharge"] = self.check_date_order(
                admissions, "admission_date", "discharge_date"
            )
            results["temporal_consistency"] = self.check_temporal_consistency(
                admissions, "admission_date", "discharge_date"
            )

        if "primary_icd_code" in admissions.columns:
            results["valid_icd"] = self.check_valid_icd(admissions, "primary_icd_code")
        if "attending_npi" in admissions.columns:
            results["valid_npi"] = self.check_valid_npi(admissions, "attending_npi")
        if "length_of_stay" in admissions.columns:
            results["los_range"] = self.check_range(admissions, "length_of_stay", min_val=0)
        if "discharge_disposition" in admissions.columns:
            valid_dispositions = self.config.get("valid_discharge_dispositions", [])
            if valid_dispositions:
                results["valid_disposition"] = self.check_values_in_set(
                    admissions, "discharge_disposition", valid_dispositions
                )

        return results

    def run_all_checks(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run all quality checks across all tables and return report.

        Expects ``data`` to contain at least 'patients' and 'admissions' keys.
        Any additional tables are logged but skipped.

        Args:
            data: Mapping of table name to DataFrame.

        Returns:
            Nested dictionary of results keyed by table name, plus a
            ``summary`` section with totals.
        """
        self.results.clear()
        report: Dict[str, Any] = {}

        if "patients" in data:
            report["patients"] = self.run_patient_checks(data["patients"])
        else:
            self.logger.warning("No 'patients' table provided.")

        if "admissions" in data:
            report["admissions"] = self.run_admission_checks(data["admissions"])
        else:
            self.logger.warning("No 'admissions' table provided.")

        for table_name, df in data.items():
            if table_name not in ("patients", "admissions"):
                self.logger.info("Skipping table '%s' (no predefined checks).", table_name)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        report["summary"] = {"total": total, "passed": passed, "failed": failed}

        self.logger.info("Quality checks complete: %d/%d passed.", passed, total)
        return report

    def generate_report(self) -> str:
        """Generate human-readable quality report.

        Returns:
            Multi-line string summarising every recorded check.
        """
        if not self.results:
            return "No quality checks have been run."

        lines: List[str] = ["=" * 60, "  Healthcare Data Quality Report", "=" * 60, ""]

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        for r in self.results:
            status = "PASS " if r["passed"] else "FAIL "
            line = f"  [{status}] {r['check']}"
            if r["details"]:
                line += f" — {r['details']}"
            lines.append(line)

        lines.extend([
            "",
            "-" * 60,
            f"  Total: {total}  |  Passed: {passed}  |  Failed: {failed}",
            "=" * 60,
        ])

        return "\n".join(lines)
