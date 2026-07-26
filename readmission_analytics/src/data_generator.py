"""Synthetic healthcare data generator for Patient Readmission Analytics."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class HealthcareDataGenerator:
    """Generate realistic synthetic healthcare data."""

    def __init__(self, n_patients: int = 5000, n_admissions: int = 20000, seed: int = 42):
        self.n_patients = n_patients
        self.n_admissions = n_admissions
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.icd10_codes = {
            "I21": ("Acute myocardial infarction", 0.08, 12000, 8),
            "I50": ("Heart failure", 0.10, 9500, 7),
            "J18": ("Pneumonia", 0.12, 8000, 6),
            "N17": ("Acute kidney injury", 0.07, 11000, 5),
            "K35": ("Acute appendicitis", 0.04, 7500, 3),
            "E11": ("Type 2 diabetes mellitus", 0.15, 5000, 5),
            "C34": ("Lung cancer", 0.03, 18000, 6),
            "J44": ("Chronic obstructive pulmonary disease", 0.08, 7000, 5),
            "I63": ("Cerebral infarction", 0.06, 13000, 8),
            "K92": ("Hematemesis", 0.03, 8500, 4),
            "M87": ("Osteonecrosis", 0.02, 6000, 4),
            "G40": ("Epilepsy", 0.04, 5500, 3),
            "J96": ("Respiratory failure", 0.05, 15000, 7),
            "K85": ("Acute pancreatitis", 0.04, 9000, 5),
            "N39": ("Urinary tract infection", 0.08, 4000, 3),
        }

        self.readmission_rates = {
            "I21": 0.25, "I50": 0.28, "J18": 0.22, "N17": 0.24,
            "E11": 0.20, "J44": 0.26, "I63": 0.23, "J96": 0.30,
            "K85": 0.18, "N39": 0.15, "C34": 0.20, "K35": 0.08,
            "K92": 0.15, "G40": 0.12, "M87": 0.10,
        }

        self.specialties = [
            ("Cardiology", 0.20), ("Pulmonology", 0.15), ("Nephrology", 0.10),
            ("Gastroenterology", 0.12), ("Oncology", 0.08), ("Neurology", 0.08),
            ("Endocrinology", 0.07), ("Orthopedics", 0.05), ("Urology", 0.05),
            ("General Surgery", 0.10),
        ]

        self.lab_tests = [
            ("CBC", "Complete Blood Count", 60),
            ("BMP", "Basic Metabolic Panel", 55),
            ("CMP", "Comprehensive Metabolic Panel", 45),
            ("Troponin", "Troponin I", 20),
            ("BNP", "B-Type Natriuretic Peptide", 18),
            ("ABG", "Arterial Blood Gas", 15),
            ("PT", "Prothrombin Time", 30),
            ("INR", "International Normalized Ratio", 30),
            ("HbA1c", "Hemoglobin A1c", 25),
            ("TSH", "Thyroid Stimulating Hormone", 22),
            ("CRP", "C-Reactive Protein", 20),
            ("Creatinine", "Serum Creatinine", 35),
            ("Lactate", "Serum Lactate", 18),
            ("D_Dimer", "D-Dimer", 15),
        ]

    def generate_patients(self) -> pd.DataFrame:
        """Generate patient records with realistic demographics."""
        patient_ids = np.arange(1, self.n_patients + 1)

        ages = self.rng.normal(loc=55, scale=18, size=self.n_patients).clip(18, 100).astype(int)

        gender = self.rng.choice(["M", "F"], size=self.n_patients, p=[0.48, 0.52])

        ethnicities = ["White", "Black", "Hispanic", "Asian", "Other"]
        ethnicity_weights = [0.55, 0.15, 0.15, 0.10, 0.05]
        ethnicity = self.rng.choice(ethnicities, size=self.n_patients, p=ethnicity_weights)

        insurance = self.rng.choice(
            ["Medicare", "Medicaid", "Private", "Self-Pay", "VA"],
            size=self.n_patients,
            p=[0.35, 0.20, 0.30, 0.10, 0.05],
        )

        comorbidities_base = ["Diabetes", "Hypertension", "COPD", "CHF", "CKD", "Obesity", "Smoking", "Depression"]
        comorbidity_probs = [0.25, 0.40, 0.15, 0.12, 0.10, 0.30, 0.20, 0.18]

        comorbidities = []
        for prob in comorbidity_probs:
            has_condition = self.rng.random(self.n_patients) < prob
            comorbidities.append(has_condition)
        comorbidity_matrix = np.column_stack(comorbidities)
        comorbidity_count = comorbidity_matrix.sum(axis=1)

        admission_counts = np.clip(
            np.round(np.exp(self.rng.normal(0.5, 0.6, size=self.n_patients))).astype(int), 1, 15
        )

        insurance_cost_map = {"Medicare": 0.8, "Medicaid": 0.7, "Private": 1.2, "Self-Pay": 0.6, "VA": 0.9}
        insurance_costs = np.array([insurance_cost_map.get(ins, 1.0) for ins in insurance])

        cost_multiplier = np.ones(self.n_patients, dtype=float)
        for i, (cond, prob) in enumerate(zip(comorbidities_base, comorbidity_probs)):
            cost_multiplier += comorbidity_matrix[:, i] * 0.15
        cost_multiplier *= insurance_costs

        patients = pd.DataFrame({
            "patient_id": patient_ids,
            "age": ages,
            "gender": gender,
            "ethnicity": ethnicity,
            "insurance_type": insurance,
            "comorbidity_count": comorbidity_count,
            "total_admissions": admission_counts,
            "cost_multiplier": np.round(cost_multiplier, 2),
        })

        return patients

    def generate_providers(self, n_providers: int = 100) -> pd.DataFrame:
        """Generate provider records with specialties."""
        provider_ids = np.arange(1, n_providers + 1)

        specialty_names = [s[0] for s in self.specialties]
        specialty_weights = [s[1] for s in self.specialties]
        provider_specialties = self.rng.choice(specialty_names, size=n_providers, p=specialty_weights)

        years_experience = self.rng.normal(loc=15, scale=8, size=n_providers).clip(1, 45).astype(int)

        base_skill = 0.7 + (years_experience / 45) * 0.25
        skill_modifier = self.rng.normal(0, 0.05, size=n_providers)
        skill_score = np.clip(base_skill + skill_modifier, 0.5, 1.0)

        first_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
                       "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez"]
        name_indices = self.rng.integers(0, len(first_names), size=n_providers)
        provider_names = [f"Dr. {first_names[i]}-{pid:03d}" for i, pid in zip(name_indices, provider_ids)]

        providers = pd.DataFrame({
            "provider_id": provider_ids,
            "name": provider_names,
            "specialty": provider_specialties,
            "years_experience": years_experience,
            "skill_score": np.round(skill_score, 3),
        })

        return providers

    def generate_diagnoses(self) -> pd.DataFrame:
        """Generate common ICD-10 diagnosis codes with descriptions."""
        diagnosis_records = []
        for code, (description, prevalence, base_cost, base_los) in self.icd10_codes.items():
            diagnosis_records.append({
                "icd10_code": code,
                "description": description,
                "prevalence": prevalence,
                "base_cost": base_cost,
                "base_los": base_los,
                "readmission_rate": self.readmission_rates.get(code, 0.15),
            })

        diagnoses = pd.DataFrame(diagnosis_records)
        return diagnoses

    def generate_admissions(
        self,
        patients: pd.DataFrame,
        providers: pd.DataFrame,
        diagnoses: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate admission records with realistic patterns."""
        admission_ids = np.arange(1, self.n_admissions + 1)

        icd_codes = list(self.icd10_codes.keys())
        code_weights = np.array([self.icd10_codes[c][1] for c in icd_codes])
        code_weights /= code_weights.sum()

        patient_sample_indices = self.rng.integers(0, len(patients), size=self.n_admissions)
        patient_ids = patients["patient_id"].values[patient_sample_indices].copy()
        patient_ages = patients["age"].values[patient_sample_indices]
        patient_multipliers = patients["cost_multiplier"].values[patient_sample_indices]

        provider_ids = self.rng.integers(1, len(providers) + 1, size=self.n_admissions)

        admission_dates = []
        base_date = datetime(2022, 1, 1)

        patient_admission_count = {}
        for pid in patient_ids:
            patient_admission_count[pid] = patient_admission_count.get(pid, 0) + 1

        patient_last_date = {}
        for i in range(self.n_admissions):
            pid = patient_ids[i]
            if pid in patient_last_date and patient_admission_count[pid] > 1:
                if self.rng.random() < 0.25:
                    gap = self.rng.integers(3, 28)
                else:
                    gap = self.rng.integers(35, 180)
                new_date = patient_last_date[pid] + timedelta(days=int(gap))
                if new_date < base_date + timedelta(days=730):
                    admission_dates.append(new_date)
                    patient_last_date[pid] = new_date
                    continue
            day_offset = self.rng.integers(0, 730)
            new_date = base_date + timedelta(days=int(day_offset))
            admission_dates.append(new_date)
            patient_last_date[pid] = new_date
        admission_dates = np.array(admission_dates)

        sort_order = np.lexsort((admission_dates, patient_ids))
        patient_ids = patient_ids[sort_order]
        patient_ages = patient_ages[sort_order]
        patient_multipliers = patient_multipliers[sort_order]
        provider_ids = provider_ids[sort_order]
        admission_dates = admission_dates[sort_order]
        admission_ids = np.arange(1, self.n_admissions + 1)

        assigned_codes = self.rng.choice(icd_codes, size=self.n_admissions, p=code_weights)

        month_of_admission = np.array([d.month for d in admission_dates])
        seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (month_of_admission - 1) / 12)
        winter_mask = (month_of_admission >= 11) | (month_of_admission <= 2)
        seasonal_factor += 0.10 * winter_mask

        los_values = np.array([self.icd10_codes[c][3] for c in assigned_codes])
        los_noise = self.rng.normal(0, 2, size=self.n_admissions)
        length_of_stay = np.clip(
            np.round(los_values + los_noise + (patient_ages - 55) * 0.05 + patient_multipliers * 0.5),
            1, 30,
        ).astype(int)

        base_costs = np.array([self.icd10_codes[c][2] for c in assigned_codes])
        los_factor = length_of_stay / np.array([self.icd10_codes[c][3] for c in assigned_codes])
        cost_noise = self.rng.normal(1.0, 0.15, size=self.n_admissions)
        admission_costs = np.clip(
            base_costs * los_factor * patient_multipliers * seasonal_factor * cost_noise,
            500, 50000,
        )

        is_readmission = np.zeros(self.n_admissions, dtype=bool)
        readmitted_within_30d = np.zeros(self.n_admissions, dtype=int)
        previous_admission_id = np.zeros(self.n_admissions, dtype=int)
        readmission_gap_days = np.zeros(self.n_admissions, dtype=int)

        for i in range(1, self.n_admissions):
            if patient_ids[i] == patient_ids[i - 1]:
                gap = (admission_dates[i] - admission_dates[i - 1]).days
                readmission_gap_days[i] = max(0, gap)
                previous_admission_id[i] = admission_ids[i - 1]
                if 0 < gap <= 30:
                    is_readmission[i] = True
                    readmitted_within_30d[i] = 1

        insurance_type = patients["insurance_type"].values[patient_sample_indices[sort_order]]

        discharge_dates = [
            admission_dates[i] + timedelta(days=int(length_of_stay[i]))
            for i in range(self.n_admissions)
        ]

        admissions = pd.DataFrame({
            "admission_id": admission_ids,
            "patient_id": patient_ids,
            "provider_id": provider_ids,
            "icd10_code": assigned_codes,
            "admission_date": admission_dates,
            "length_of_stay": length_of_stay,
            "discharge_date": discharge_dates,
            "cost": np.round(admission_costs, 2),
            "seasonal_factor": np.round(seasonal_factor, 3),
            "is_readmission": is_readmission.astype(int),
            "readmitted_within_30d": readmitted_within_30d,
            "previous_admission_id": previous_admission_id,
            "readmission_gap_days": readmission_gap_days,
            "insurance_type": insurance_type,
        })

        readmission_rate = admissions["readmitted_within_30d"].mean()
        print(f"Generated {self.n_admissions} admissions | 30-day readmission rate: {readmission_rate:.1%}")

        return admissions

    def generate_procedures(self, admissions: pd.DataFrame) -> pd.DataFrame:
        """Generate procedure records for admissions."""
        procedure_defs = [
            ("99213", "Office Visit", 150),
            ("99214", "Office Visit - Detailed", 250),
            ("99223", "Inpatient Initial", 400),
            ("99232", "Subsequent Hospital Care", 200),
            ("36556", "Central Venous Catheter", 1500),
            ("93000", "Electrocardiogram", 100),
            ("71046", "Chest X-Ray", 150),
            ("70553", "Brain MRI", 2500),
            ("93306", "Echocardiography", 800),
            ("43239", "Upper GI Endoscopy", 3000),
            ("49505", "Inguinal Hernia Repair", 5000),
            ("27447", "Knee Replacement", 25000),
            ("85025", "CBC with Differential", 80),
            ("80053", "Comprehensive Metabolic Panel", 75),
            ("84443", "TSH Level", 65),
        ]

        n_procedures = int(self.n_admissions * self.rng.uniform(1.5, 3.0))
        procedure_ids = np.arange(1, n_procedures + 1)

        admission_indices = self.rng.integers(0, len(admissions), size=n_procedures)
        admission_id_vals = admissions["admission_id"].values[admission_indices]
        patient_id_vals = admissions["patient_id"].values[admission_indices]

        proc_indices = self.rng.integers(0, len(procedure_defs), size=n_procedures)
        procedure_codes = [procedure_defs[i][0] for i in proc_indices]
        procedure_names = [procedure_defs[i][1] for i in proc_indices]
        base_costs = np.array([procedure_defs[i][2] for i in proc_indices])

        cost_variation = self.rng.normal(1.0, 0.1, size=n_procedures)
        procedure_costs = np.round(base_costs * cost_variation, 2)

        procedure_dates = []
        admission_dates_vals = admissions["admission_date"].values[admission_indices]
        for adate in admission_dates_vals:
            offset = self.rng.integers(0, 3)
            procedure_dates.append(pd.Timestamp(adate) + timedelta(days=int(offset)))

        procedures = pd.DataFrame({
            "procedure_id": procedure_ids,
            "admission_id": admission_id_vals,
            "patient_id": patient_id_vals,
            "procedure_code": procedure_codes,
            "procedure_name": procedure_names,
            "procedure_date": procedure_dates,
            "cost": procedure_costs,
        })

        print(f"Generated {n_procedures} procedure records")
        return procedures

    def generate_lab_results(self, admissions: pd.DataFrame) -> pd.DataFrame:
        """Generate lab results with abnormal flags."""
        n_labs = int(self.n_admissions * self.rng.uniform(3.0, 6.0))
        lab_ids = np.arange(1, n_labs + 1)

        admission_indices = self.rng.integers(0, len(admissions), size=n_labs)
        admission_id_vals = admissions["admission_id"].values[admission_indices]
        patient_id_vals = admissions["patient_id"].values[admission_indices]

        test_indices = self.rng.integers(0, len(self.lab_tests), size=n_labs)
        test_names = [self.lab_tests[i][0] for i in test_indices]
        test_full_names = [self.lab_tests[i][1] for i in test_indices]
        base_costs = np.array([self.lab_tests[i][2] for i in test_indices])
        lab_costs = np.round(base_costs * self.rng.normal(1.0, 0.08, size=n_labs), 2)

        lab_dates = []
        admission_dates_vals = admissions["admission_date"].values[admission_indices]
        for adate in admission_dates_vals:
            offset = self.rng.integers(0, 5)
            lab_dates.append(pd.Timestamp(adate) + timedelta(days=int(offset)))

        normal_rate = self.rng.uniform(0.6, 0.85, size=n_labs)
        abnormal_flag = (self.rng.random(n_labs) > normal_rate).astype(int)
        abnormal_labels = np.where(abnormal_flag, "Abnormal", "Normal")

        is_critical = np.zeros(n_labs, dtype=int)
        critical_mask = abnormal_flag.astype(bool) & (self.rng.random(n_labs) < 0.15)
        is_critical[critical_mask] = 1

        reference_ranges = {
            "CBC": (4.5, 11.0), "BMP": (135, 145), "CMP": (70, 100),
            "Troponin": (0.0, 0.04), "BNP": (0, 100), "ABG": (7.35, 7.45),
            "PT": (11, 13.5), "INR": (0.8, 1.2), "HbA1c": (4.0, 5.7),
            "TSH": (0.4, 4.0), "CRP": (0.0, 3.0), "Creatinine": (0.6, 1.2),
            "Lactate": (0.5, 2.0), "D_Dimer": (0.0, 0.5),
        }

        result_values = []
        for i in range(n_labs):
            test = test_names[i]
            low, high = reference_ranges.get(test, (0, 100))
            ref_range = f"{low}-{high}"
            if abnormal_flag[i]:
                if self.rng.random() < 0.5:
                    val = self.rng.uniform(low * 0.5, low)
                else:
                    val = self.rng.uniform(high, high * 1.5)
            else:
                val = self.rng.uniform(low, high)
            result_values.append(round(float(val), 2))

        lab_results = pd.DataFrame({
            "lab_id": lab_ids,
            "admission_id": admission_id_vals,
            "patient_id": patient_id_vals,
            "test_name": test_names,
            "test_full_name": test_full_names,
            "result_value": result_values,
            "reference_range": [reference_ranges.get(t, (0, 100)) for t in test_names],
            "abnormal_flag": abnormal_labels,
            "is_critical": is_critical,
            "lab_date": lab_dates,
            "cost": lab_costs,
        })

        abnormal_rate = lab_results["abnormal_flag"].eq("Abnormal").mean()
        print(f"Generated {n_labs} lab results | Abnormal rate: {abnormal_rate:.1%}")
        return lab_results

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all datasets and return as dict."""
        print("=" * 60)
        print("Generating Synthetic Healthcare Data")
        print("=" * 60)

        patients = self.generate_patients()
        print(f"Generated {len(patients)} patient records")

        providers = self.generate_providers(n_providers=100)
        print(f"Generated {len(providers)} provider records")

        diagnoses = self.generate_diagnoses()
        print(f"Generated {len(diagnoses)} diagnosis codes")

        admissions = self.generate_admissions(patients, providers, diagnoses)

        procedures = self.generate_procedures(admissions)

        lab_results = self.generate_lab_results(admissions)

        print("=" * 60)
        print("Data Generation Complete")
        print("=" * 60)

        return {
            "patients": patients,
            "providers": providers,
            "diagnoses": diagnoses,
            "admissions": admissions,
            "procedures": procedures,
            "lab_results": lab_results,
        }


if __name__ == "__main__":
    generator = HealthcareDataGenerator(n_patients=5000, n_admissions=20000, seed=42)
    datasets = generator.generate_all()

    print("\nDataset shapes:")
    for name, df in datasets.items():
        print(f"  {name}: {df.shape}")

    print("\nAdmission cost summary:")
    print(datasets["admissions"]["cost"].describe())
