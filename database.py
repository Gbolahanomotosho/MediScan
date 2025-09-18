# database.py - MediScan Database Schema
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime
import os
import json

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
    board_certifications = Column(Text, nullable=True)  # JSON string
    
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
    patient_id = Column(Integer, nullable=False, index=True)
    doctor_id = Column(Integer, nullable=False, index=True)
    
    # Image information
    original_filename = Column(String(255), nullable=False)
    image_size = Column(Integer, nullable=True)  # File size in bytes
    image_format = Column(String(20), nullable=True)  # JPEG, PNG, etc.
    image_dimensions = Column(String(50), nullable=True)  # "1024x1024"
    
    # Analysis results
    analysis_status = Column(String(50), default="pending")  # pending, completed, failed
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
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
    urgency_level = Column(String(20), default="routine")  # routine, urgent, emergency
    
    # Report generation
    pdf_generated = Column(Boolean, default=False)
    pdf_path = Column(String(500), nullable=True)
    email_sent = Column(Boolean, default=False)
    email_sent_at = Column(DateTime(timezone=True), nullable=True)
    
    # Quality metrics
    image_quality_score = Column(Float, nullable=True)  # 0.0 to 1.0
    positioning_quality = Column(String(50), nullable=True)  # excellent, good, fair, poor
    
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
    diagnosis_id = Column(Integer, nullable=False, index=True)
    
    # Condition information
    condition_name = Column(String(255), nullable=False, index=True)
    condition_category = Column(String(100), nullable=True)  # lung, heart, bone, etc.
    confidence_score = Column(Float, nullable=False)
    severity_level = Column(String(50), nullable=True)  # mild, moderate, severe
    
    # Location information (if available)
    lung_region = Column(String(100), nullable=True)  # left_upper, right_lower, etc.
    anatomical_location = Column(String(255), nullable=True)
    bounding_box = Column(String(255), nullable=True)  # "x1,y1,x2,y2"
    
    # Clinical significance
    is_primary_finding = Column(Boolean, default=False)
    clinical_significance = Column(String(100), nullable=True)  # significant, incidental, artifact
    requires_followup = Column(Boolean, default=False)
    
    # Additional metadata
    gradcam_generated = Column(Boolean, default=False)
    gradcam_intensity = Column(Float, nullable=True)  # Average heatmap intensity
    
    # ICD-10 coding (if available)
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
    metric_type = Column(String(50), nullable=False)  # performance, usage, accuracy, etc.
    
    # Context
    diagnosis_id = Column(Integer, nullable=True, index=True)
    model_version = Column(String(50), nullable=True)
    session_id = Column(String(100), nullable=True)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
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
    user_type = Column(String(50), nullable=True)  # doctor, patient, system, admin
    user_identifier = Column(String(255), nullable=True)  # email or ID
    
    # Context
    diagnosis_id = Column(Integer, nullable=True, index=True)
    patient_id = Column(Integer, nullable=True, index=True)
    doctor_id = Column(Integer, nullable=True, index=True)
    
    # Request information
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    session_id = Column(String(100), nullable=True)
    
    # Result information
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)  # Activity duration in milliseconds
    
    # Additional data
    additional_data = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<AuditLog(activity='{self.activity_type}', success={self.success})>"

# Database utility functions
def init_db():
    """Initialize database with all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ MediScan database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_patient(db: SessionLocal, patient_data: dict) -> Patient:
    """Create a new patient record"""
    try:
        patient = Patient(**patient_data)
        db.add(patient)
        db.commit()
        db.refresh(patient)
        return patient
    except Exception as e:
        db.rollback()
        raise e

def create_doctor(db: SessionLocal, doctor_data: dict) -> Doctor:
    """Create a new doctor record"""
    try:
        doctor = Doctor(**doctor_data)
        db.add(doctor)
        db.commit()
        db.refresh(doctor)
        return doctor
    except Exception as e:
        db.rollback()
        raise e

def create_diagnosis(db: SessionLocal, diagnosis_data: dict) -> Diagnosis:
    """Create a new diagnosis record"""
    try:
        diagnosis = Diagnosis(**diagnosis_data)
        db.add(diagnosis)
        db.commit()
        db.refresh(diagnosis)
        return diagnosis
    except Exception as e:
        db.rollback()
        raise e

def add_findings(db: SessionLocal, diagnosis_id: int, findings_data: list) -> list:
    """Add findings to a diagnosis"""
    try:
        findings = []
        for finding_data in findings_data:
            finding_data['diagnosis_id'] = diagnosis_id
            finding = Finding(**finding_data)
            db.add(finding)
            findings.append(finding)
        
        db.commit()
        return findings
    except Exception as e:
        db.rollback()
        raise e

def log_activity(db: SessionLocal, activity_type: str, description: str, **kwargs):
    """Log system activity"""
    try:
        audit_log = AuditLog(
            activity_type=activity_type,
            activity_description=description,
            **kwargs
        )
        db.add(audit_log)
        db.commit()
    except Exception as e:
        print(f"Failed to log activity: {e}")

def record_metric(db: SessionLocal, metric_name: str, value: float, metric_type: str, **kwargs):
    """Record system metric"""
    try:
        metric = SystemMetrics(
            metric_name=metric_name,
            metric_value=value,
            metric_type=metric_type,
            **kwargs
        )
        db.add(metric)
        db.commit()
    except Exception as e:
        print(f"Failed to record metric: {e}")

def get_diagnosis_by_report_id(db: SessionLocal, report_id: str) -> Diagnosis:
    """Get diagnosis by report ID"""
    return db.query(Diagnosis).filter(Diagnosis.report_id == report_id).first()

def get_patient_by_email(db: SessionLocal, email: str) -> Patient:
    """Get patient by email"""
    return db.query(Patient).filter(Patient.email == email).first()

def get_doctor_by_name(db: SessionLocal, name: str) -> Doctor:
    """Get doctor by name"""
    return db.query(Doctor).filter(Doctor.name == name).first()

def get_recent_diagnoses(db: SessionLocal, limit: int = 100) -> list:
    """Get recent diagnoses"""
    return db.query(Diagnosis).order_by(Diagnosis.created_at.desc()).limit(limit).all()

def get_diagnosis_stats(db: SessionLocal, days: int = 30) -> dict:
    """Get diagnosis statistics"""
    try:
        from datetime import datetime, timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        total_diagnoses = db.query(Diagnosis).filter(Diagnosis.created_at >= cutoff_date).count()
        completed_diagnoses = db.query(Diagnosis).filter(
            Diagnosis.created_at >= cutoff_date,
            Diagnosis.analysis_status == "completed"
        ).count()
        
        abnormal_cases = db.query(Diagnosis).filter(
            Diagnosis.created_at >= cutoff_date,
            Diagnosis.has_abnormalities == True
        ).count()
        
        # Most common findings
        common_findings = db.query(Finding.condition_name, func.count(Finding.id))\
            .join(Diagnosis)\
            .filter(Diagnosis.created_at >= cutoff_date)\
            .group_by(Finding.condition_name)\
            .order_by(func.count(Finding.id).desc())\
            .limit(10).all()
        
        return {
            "total_diagnoses": total_diagnoses,
            "completed_diagnoses": completed_diagnoses,
            "abnormal_cases": abnormal_cases,
            "success_rate": (completed_diagnoses / total_diagnoses * 100) if total_diagnoses > 0 else 0,
            "abnormality_rate": (abnormal_cases / completed_diagnoses * 100) if completed_diagnoses > 0 else 0,
            "common_findings": [{"condition": name, "count": count} for name, count in common_findings]
        }
    except Exception as e:
        print(f"Failed to get statistics: {e}")
        return {}

def check_database_health() -> dict:
    """Check database health"""
    try:
        db = SessionLocal()
        
        # Test basic queries
        patient_count = db.query(Patient).count()
        doctor_count = db.query(Doctor).count()
        diagnosis_count = db.query(Diagnosis).count()
        
        db.close()
        
        return {
            "status": "healthy",
            "connection": "ok",
            "table_counts": {
                "patients": patient_count,
                "doctors": doctor_count,
                "diagnoses": diagnosis_count
            },
            "last_checked": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connection": "failed",
            "error": str(e),
            "last_checked": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    # Initialize database when run directly
    if init_db():
        print("✅ MediScan database setup completed successfully")
        
        # Print health check
        health = check_database_health()
        print(f"Database health: {health}")
    else:
        print("❌ MediScan database setup failed")
