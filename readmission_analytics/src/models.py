from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Date, Numeric, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Patient(Base):
    """Patient dimension table."""
    __tablename__ = 'patients'
    
    patient_id = Column(Integer, primary_key=True, autoincrement=True)
    mrn = Column(String(20), nullable=False, unique=True)  # Medical Record Number
    name = Column(String(100))
    date_of_birth = Column(Date)
    gender = Column(String(10))
    race = Column(String(50))
    ethnicity = Column(String(50))
    insurance_type = Column(String(30))  # Medicare, Medicaid, Private, Self-Pay
    zip_code = Column(String(10))
    admission_date = Column(DateTime)
    discharge_date = Column(DateTime)
    length_of_stay = Column(Integer)
    
class Admission(Base):
    """Admission/fact table for hospital visits."""
    __tablename__ = 'admissions'
    
    admission_id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.patient_id'), nullable=False)
    provider_id = Column(Integer, ForeignKey('providers.provider_id'))
    admission_date = Column(DateTime, nullable=False)
    discharge_date = Column(DateTime)
    admission_type = Column(String(30))  # Emergency, Elective, Urgent, Newborn
    discharge_disposition = Column(String(50))  # Home, Transfer, Expired, AMA
    admission_source = Column(String(50))  # Physician Referral, Emergency, Transfer
    primary_diagnosis_code = Column(String(10))
    secondary_diagnosis_code = Column(String(10))
    total_charges = Column(Numeric(12, 2))
    total_payments = Column(Numeric(12, 2))
    insurance_payments = Column(Numeric(12, 2))
    patient_payments = Column(Numeric(12, 2))
    medication_charges = Column(Numeric(12, 2))
    lab_charges = Column(Numeric(12, 2))
    procedure_charges = Column(Numeric(12, 2))
    readmitted_30d = Column(Boolean, default=False)
    days_to_readmission = Column(Integer)
    
    __table_args__ = (
        Index('idx_admission_patient', 'patient_id'),
        Index('idx_admission_date', 'admission_date'),
        Index('idx_admission_provider', 'provider_id'),
    )

class Diagnosis(Base):
    """Diagnosis dimension table."""
    __tablename__ = 'diagnoses'
    
    diagnosis_id = Column(Integer, primary_key=True, autoincrement=True)
    icd_code = Column(String(10), nullable=False)
    icd_version = Column(Integer, default=10)  # ICD-9 or ICD-10
    description = Column(String(200))
    category = Column(String(100))  # Circulatory, Respiratory, Digestive, etc.
    chapter = Column(String(100))  # ICD chapter
    
class Provider(Base):
    """Provider/physician dimension table."""
    __tablename__ = 'providers'
    
    provider_id = Column(Integer, primary_key=True, autoincrement=True)
    npi = Column(String(20), unique=True)  # National Provider Identifier
    name = Column(String(100))
    specialty = Column(String(50))
    department = Column(String(50))
    hospital = Column(String(100))

class Procedure(Base):
    """Procedure fact table."""
    __tablename__ = 'procedures'
    
    procedure_id = Column(Integer, primary_key=True, autoincrement=True)
    admission_id = Column(Integer, ForeignKey('admissions.admission_id'))
    procedure_code = Column(String(10))
    procedure_date = Column(DateTime)
    charges = Column(Numeric(12, 2))
    
class LabResult(Base):
    """Lab result fact table."""
    __tablename__ = 'lab_results'
    
    lab_id = Column(Integer, primary_key=True, autoincrement=True)
    admission_id = Column(Integer, ForeignKey('admissions.admission_id'))
    lab_test = Column(String(50))
    result_value = Column(String(50))
    result_date = Column(DateTime)
    is_abnormal = Column(Boolean, default=False)
