# database.py - MediScan Database Schema (FIXED with ForeignKeys)
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Text, Boolean, JSON, LargeBinary, ForeignKey
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mediscan.db")

# For SQLite, add connection args. For PostgreSQL, remove this.
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Patient(Base):
    """Patient information model"""
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    email = Column(String(255), nullable=True, index=True)
    phone = Column(String(50), nullable=True)
    
    # Medical information
    medical_history = Column(Text, nullable=True)
    allergies = Column(Text, nullable=True)
    current_medications = Column(Text, nullable=True)
    
    # Emergency contact
    emergency_contact_name = Column(String(255), nullable=True)
    emergency_contact_phone = Column(String(50), nullable=True)
    
    # Address information
    address = Column(Text, nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    zip_code = Column(String(20), nullable=True)
    country = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    diagnoses = relationship("Diagnosis", back_populates="patient", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Patient(name='{self.name}', age={self.age})>"

class Doctor(Base):
    """Doctor information model"""
    __tablename__ = "doctors"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    specialization = Column(String(255), nullable=True)
    license_number = Column(String(100), nullable=True, unique=True)
    email = Column(String(255), nullable=True, index=True)
    phone = Column(String(50), nullable=True)
    
    # Professional information
    hospital_affiliation = Column(String(255), nullable=True)
    department = Column(String(255), nullable=True)
    years_of_experience = Column(Integer, nullable=True)
    
    # Credentials
    medical_school = Column(String(255), nullable=True)
    residency = Column(String(255), nullable=True)
    board_certifications = Column(Text, nullable=True)
    
    # Contact information
    office_address = Column(Text, nullable=True)
    office_phone = Column(String(50), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    diagnoses = relationship("Diagnosis", back_populates="doctor", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Doctor(name='{self.name}', specialization='{self.specialization}')>"

class Diagnosis(Base):
    """Main diagnosis record model"""
    __tablename__ = "diagnoses"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(100), unique=True, nullable=False, index=True)
    
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False, index=True)
    
    # Image information
    original_filename = Column(String(255), nullable=False)
    image_size = Column(Integer, nullable=True)
    image_format = Column(String(20), nullable=True)
    image_dimensions = Column(String(50), nullable=True)
    
    # Analysis results
    analysis_status = Column(String(50), default="pending")
    processing_time = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    confidence_threshold = Column(Float, default=0.5)
    
    # Findings summary
    has_abnormalities = Column(Boolean, default=False)
    abnormalities_count = Column(Integer, default=0)
    max_confidence = Column(Float, nullable=True)
    primary_finding = Column(String(255), nullable=True)
    
    # Clinical notes
    clinical_notes = Column(Text, nullable=True)
    doctor_interpretation = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    follow_up_required = Column(Boolean, default=False)
    urgency_level = Column(String(20), default="routine")
    
    # Report generation
    pdf_generated = Column(Boolean, default=False)
    pdf_path = Column(String(500), nullable=True)
    email_sent = Column(Boolean, default=False)
    email_sent_at = Column(DateTime(timezone=True), nullable=True)
    
    # Quality metrics
    image_quality_score = Column(Float, nullable=True)
    positioning_quality = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    analyzed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    patient = relationship("Patient", back_populates="diagnoses")
    doctor = relationship("Doctor", back_populates="diagnoses")
    findings = relationship("Finding", back_populates="diagnosis", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Diagnosis(report_id='{self.report_id}', status='{self.analysis_status}')>"

class Finding(Base):
    """Individual finding/condition detected"""
    __tablename__ = "findings"
    
    id = Column(Integer, primary_key=True, index=True)
    diagnosis_id = Column(Integer, ForeignKey("diagnoses.id"), nullable=False, index=True)
    
    # Condition information
    condition_name = Column(String(255), nullable=False, index=True)
    condition_category = Column(String(100), nullable=True)
    confidence_score = Column(Float, nullable=False)
    severity_level = Column(String(50), nullable=True)
    
    # Location information
    lung_region = Column(String(100), nullable=True)
    anatomical_location = Column(String(255), nullable=True)
    bounding_box = Column(String(255), nullable=True)
    
    # Clinical significance
    is_primary_finding = Column(Boolean, default=False)
    clinical_significance = Column(String(100), nullable=True)
    requires_followup = Column(Boolean, default=False)
    
    # Additional metadata
    gradcam_generated = Column(Boolean, default=False)
    gradcam_intensity = Column(Float, nullable=True)
    
    # ICD-10 coding
    icd10_code = Column(String(20), nullable=True)
    icd10_description = Column(String(500), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    diagnosis = relationship("Diagnosis", back_populates="findings")
    
    def __repr__(self):
        return f"<Finding(condition='{self.condition_name}', confidence={self.confidence_score})>"

class SystemMetrics(Base):
    """System performance and usage metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Metric information
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50), nullable=True)
    metric_type = Column(String(50), nullable=False)
    
    # Context
    diagnosis_id = Column(Integer, ForeignKey("diagnoses.id"), nullable=True, index=True)
    model_version = Column(String(50), nullable=True)
    session_id = Column(String(100), nullable=True)
    
    # Additional data
    additional_metadata = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Timestamps
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<SystemMetrics(name='{self.metric_name}', value={self.metric_value})>"

class AuditLog(Base):
    """Audit trail for system activities"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Activity information
    activity_type = Column(String(100), nullable=False, index=True)
    activity_description = Column(String(500), nullable=False)
    user_type = Column(String(50), nullable=True)
    user_identifier = Column(String(255), nullable=True)
    
    # Context
    diagnosis_id = Column(Integer, ForeignKey("diagnoses.id"), nullable=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True, index=True)
    
    # Request information
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    session_id = Column(String(100), nullable=True)
    
    # Result information
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    
    # Additional data
    extra_data = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<AuditLog(activity='{self.activity_type}', success={self.success})>"

# --- Utility functions (unchanged) ---

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ MediScan database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
