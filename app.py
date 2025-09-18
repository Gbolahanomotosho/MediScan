# app.py - Enhanced MediScan FastAPI Application
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
from starlette.requests import Request
from sqlalchemy.orm import Session

import os
import uuid
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict
import asyncio
from PIL import Image
import io
from pathlib import Path

# Import our modules
from analyze import analyze_xray, check_model_availability, get_model_info
from pdf_report import generate_pdf_report
from email_sender import send_email
from database import (
    get_db, init_db, Patient, Doctor, Diagnosis, Finding, 
    create_patient, create_doctor, create_diagnosis, add_findings,
    log_activity, record_metric, get_diagnosis_by_report_id,
    get_patient_by_email, get_doctor_by_name, check_database_health
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the port from environment variable or default to 8000
PORT = int(os.environ.get("PORT", 8000))

# Initialize FastAPI app
app = FastAPI(
    title="MediScan AI - Advanced Chest X-ray Diagnosis System",
    description="AI-powered chest X-ray analysis for detecting 14 thoracic conditions",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories and templates
os.makedirs("reports", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
MAX_FILE_SIZE = 300 * 1024 * 1024  # 300MB limit
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
MIN_IMAGE_SIZE = (224, 224)

# Initialize database
init_db()

class SecurityValidator:
    """Security validation for uploaded files"""
    
    @staticmethod
    def validate_file_type(filename: str, content_type: str) -> bool:
        """Validate file type"""
        if not content_type.startswith("image/"):
            return False
        
        ext = os.path.splitext(filename.lower())[1]
        return ext in ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_file_size(content: bytes) -> bool:
        """Validate file size"""
        return len(content) <= MAX_FILE_SIZE
    
    @staticmethod
    def validate_image_content(content: bytes) -> tuple:
        """Validate actual image content"""
        try:
            image = Image.open(io.BytesIO(content))
            width, height = image.size
            
            # Check minimum dimensions
            if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                return False, f"Image too small: {width}x{height}, minimum required: {MIN_IMAGE_SIZE[0]}x{MIN_IMAGE_SIZE[1]}"
            
            # Check if image is corrupted
            image.verify()
            
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting MediScan AI Application v2.0.0")
    
    # Check model availability
    if not check_model_availability():
        logger.warning("⚠️  Model file not found. Please ensure 'model.pth.tar' is available.")
    
    # Check database health
    health = check_database_health()
    if health['status'] == 'healthy':
        logger.info("✅ Database connection healthy")
    else:
        logger.error(f"❌ Database connection failed: {health.get('error', 'Unknown error')}")

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Main upload form page"""
    model_info = get_model_info()
    return templates.TemplateResponse("form.html", {
        "request": request,
        "model_info": model_info,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "supported_formats": list(ALLOWED_EXTENSIONS)
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_available = check_model_availability()
    db_health = check_database_health()
    
    return {
        "status": "healthy" if model_available and db_health['status'] == 'healthy' else "degraded",
        "model_available": model_available,
        "database": db_health,
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/analyze/")
async def analyze_xray_endpoint(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    doctor_name: str = Form(...),
    patient_name: str = Form(...),
    patient_age: int = Form(...),
    patient_gender: str = Form(...),
    email: str = Form(...),
    clinical_notes: Optional[str] = Form(""),
    urgency_level: Optional[str] = Form("routine"),
    db: Session = Depends(get_db)
):
    """
    Analyze uploaded chest X-ray image
    """
    start_time = time.time()
    report_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting analysis for report {report_id}")
        
        # Log activity
        log_activity(
            db, "analysis_started", 
            f"X-ray analysis initiated for patient {patient_name}",
            user_type="doctor",
            user_identifier=doctor_name
        )
        
        # Validate input parameters
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if patient_age < 0 or patient_age > 150:
            raise HTTPException(status_code=400, detail="Invalid age")
        
        if patient_gender.lower() not in ['male', 'female', 'other']:
            raise HTTPException(status_code=400, detail="Invalid gender")
        
        # Read file content
        try:
            contents = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
        
        # Security validation
        validator = SecurityValidator()
        
        # Validate file type
        if not validator.validate_file_type(file.filename, file.content_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Validate file size
        if not validator.validate_file_size(contents):
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Validate image content
        is_valid, validation_message = validator.validate_image_content(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_message)
        
        logger.info(f"File validation passed for {file.filename}")
        
        # Get or create patient
        patient = get_patient_by_email(db, email)
        if not patient:
            patient_data = {
                'name': patient_name,
                'age': patient_age,
                'gender': patient_gender,
                'email': email
            }
            patient = create_patient(db, patient_data)
            logger.info(f"Created new patient record: {patient.id}")
        
        # Get or create doctor
        doctor = get_doctor_by_name(db, doctor_name)
        if not doctor:
            doctor_data = {
                'name': doctor_name,
                'specialization': 'Radiologist'
            }
            doctor = create_doctor(db, doctor_data)
            logger.info(f"Created new doctor record: {doctor.id}")
        
        # Create initial diagnosis record
        diagnosis_data = {
            'report_id': report_id,
            'patient_id': patient.id,
            'doctor_id': doctor.id,
            'original_filename': file.filename,
            'image_size': len(contents),
            'image_format': file.content_type.split('/')[-1].upper(),
            'clinical_notes': clinical_notes,
            'urgency_level': urgency_level,
            'analysis_status': 'processing'
        }
        
        # Get image dimensions
        try:
            image = Image.open(io.BytesIO(contents))
            diagnosis_data['image_dimensions'] = f"{image.size[0]}x{image.size[1]}"
        except:
            pass
        
        diagnosis = create_diagnosis(db, diagnosis_data)
        logger.info(f"Created diagnosis record: {diagnosis.id}")
        
        # Perform AI analysis
        logger.info("Starting AI analysis...")
        findings, heatmap_bytes, metadata = analyze_xray(contents, threshold=0.4, generate_heatmap=True)
        
        analysis_time = time.time() - start_time
        logger.info(f"AI analysis completed in {analysis_time:.2f}s")
        
        # Update diagnosis with results
        diagnosis.analysis_status = 'completed'
        diagnosis.processing_time = analysis_time
        diagnosis.has_abnormalities = metadata.get('has_abnormalities', False)
        diagnosis.max_confidence = metadata.get('max_confidence', 0.0)
        diagnosis.analyzed_at = datetime.utcnow()
        diagnosis.image_quality_score = metadata.get('image_quality_score', 0.0)
        
        # Determine primary finding
        if metadata.get('has_abnormalities'):
            # Find highest confidence finding from the text report
            max_conf = 0
            primary_finding = "Multiple findings detected"
            for finding_text in findings:
                if ":" in finding_text and "probability" in finding_text:
                    try:
                        condition = finding_text.split(":")[0].strip().replace("•", "").strip()
                        conf_str = finding_text.split(":")[1].strip()
                        if "probability" in conf_str:
                            conf_val = float(conf_str.replace("probability", "").strip())
                            if conf_val > max_conf:
                                max_conf = conf_val
                                primary_finding = condition
                    except:
                        continue
            diagnosis.primary_finding = primary_finding
        
        # Count abnormalities
        abnormality_count = sum(1 for f in findings if "probability" in f and ":" in f)
        diagnosis.abnormalities_count = abnormality_count
        
        db.commit()
        
        # Process findings for database storage
        findings_data = []
        for finding_text in findings:
            if ":" in finding_text and "probability" in finding_text:
                try:
                    parts = finding_text.split(":")
                    condition_name = parts[0].strip().replace("•", "").strip()
                    confidence_str = parts[1].strip()
                    confidence = float(confidence_str.replace("probability", "").strip())
                    
                    finding_data = {
                        'condition_name': condition_name,
                        'confidence_score': confidence,
                        'is_primary_finding': confidence == metadata.get('max_confidence', 0.0),
                        'gradcam_generated': heatmap_bytes is not None
                    }
                    findings_data.append(finding_data)
                except:
                    continue
        
        # Add findings to database
        if findings_data:
            add_findings(db, diagnosis.id, findings_data)
        
        # Save heatmap image
        heatmap_path = None
        if heatmap_bytes:
            heatmap_filename = f"{report_id}_heatmap.png"
            heatmap_path = f"static/{heatmap_filename}"
            
            try:
                with open(heatmap_path, "wb") as f:
                    f.write(heatmap_bytes)
                logger.info(f"Heatmap saved: {heatmap_path}")
            except Exception as e:
                logger.error(f"Failed to save heatmap: {e}")
                heatmap_path = None
        
        # Generate PDF report in background
        pdf_path = f"reports/{report_id}.pdf"
        background_tasks.add_task(
            generate_and_send_report,
            doctor_name, patient_name, patient_age, patient_gender,
            findings, heatmap_bytes, pdf_path, email, report_id, db
        )
        
        # Record metrics
        record_metric(db, "analysis_time", analysis_time, "performance", diagnosis_id=diagnosis.id)
        record_metric(db, "image_quality", metadata.get('image_quality_score', 0), "quality", diagnosis_id=diagnosis.id)
        record_metric(db, "findings_count", len(findings_data), "analysis", diagnosis_id=diagnosis.id)
        
        # Log completion
        log_activity(
            db, "analysis_completed",
            f"X-ray analysis completed for {patient_name}. Found {len(findings_data)} findings.",
            user_type="doctor",
            user_identifier=doctor_name,
            diagnosis_id=diagnosis.id
        )
        
        # Prepare response data
        response_data = {
            "request": request,
            "report_id": report_id,
            "doctor_name": doctor_name,
            "patient_name": patient_name,
            "patient_age": patient_age,
            "patient_gender": patient_gender,
            "findings": findings,
            "heatmap_url": f"/static/{report_id}_heatmap.png" if heatmap_path else None,
            "pdf_url": f"/download/{report_id}",
            "metadata": metadata,
            "processing_time": f"{analysis_time:.2f}",
            "abnormalities_detected": metadata.get('has_abnormalities', False),
            "confidence_score": f"{metadata.get('max_confidence', 0) * 100:.1f}%",
            "image_quality": f"{metadata.get('image_quality_score', 0) * 100:.1f}%"
        }
        
        return templates.TemplateResponse("report.html", response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for report {report_id}: {str(e)}")
        
        # Update diagnosis status
        try:
            diagnosis = get_diagnosis_by_report_id(db, report_id)
            if diagnosis:
                diagnosis.analysis_status = 'failed'
                db.commit()
        except:
            pass
        
        # Log error
        log_activity(
            db, "analysis_failed",
            f"X-ray analysis failed: {str(e)}",
            user_type="doctor",
            user_identifier=doctor_name,
            success=False,
            error_message=str(e)
        )
        
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def generate_and_send_report(
    doctor_name: str,
    patient_name: str, 
    patient_age: int,
    patient_gender: str,
    findings: List[str],
    heatmap_bytes: bytes,
    pdf_path: str,
    email: str,
    report_id: str,
    db: Session
):
    """Background task to generate and send PDF report"""
    try:
        logger.info(f"Generating PDF report for {report_id}")
        
        # Generate PDF
        generate_pdf_report(
            doctor_name, patient_name, patient_age, patient_gender,
            findings, heatmap_bytes, output_path=pdf_path
        )
        
        # Update diagnosis
        diagnosis = get_diagnosis_by_report_id(db, report_id)
        if diagnosis:
            diagnosis.pdf_generated = True
            diagnosis.pdf_path = pdf_path
            db.commit()
        
        logger.info(f"PDF generated successfully: {pdf_path}")
        
        # Send email
        try:
            send_email(email, pdf_path)
            
            # Update diagnosis
            if diagnosis:
                diagnosis.email_sent = True
                diagnosis.email_sent_at = datetime.utcnow()
                db.commit()
            
            logger.info(f"Email sent successfully to {email}")
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
        
        # Log activity
        log_activity(
            db, "report_generated",
            f"PDF report generated and sent for {patient_name}",
            diagnosis_id=diagnosis.id if diagnosis else None
        )
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")

@app.get("/download/{report_id}")
async def download_pdf(report_id: str, db: Session = Depends(get_db)):
    """Download PDF report"""
    try:
        diagnosis = get_diagnosis_by_report_id(db, report_id)
        if not diagnosis:
            raise HTTPException(status_code=404, detail="Report not found")
        
        pdf_path = f"reports/{report_id}.pdf"
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        # Log download
        log_activity(
            db, "report_downloaded",
            f"PDF report downloaded for report {report_id}",
            diagnosis_id=diagnosis.id
        )
        
        return FileResponse(
            pdf_path, 
            media_type='application/pdf', 
            filename=f"MediScan_Report_{report_id}.pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail="Download failed")

@app.get("/report/{report_id}")
async def view_report(report_id: str, request: Request, db: Session = Depends(get_db)):
    """View existing report"""
    try:
        diagnosis = get_diagnosis_by_report_id(db, report_id)
        if not diagnosis:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Reconstruct findings from database
        findings = []
        for finding in diagnosis.findings:
            findings.append(f"{finding.condition_name}: {finding.confidence_score:.2f} probability")
        
        if not findings:
            findings = ["Analysis results not available"]
        
        response_data = {
            "request": request,
            "report_id": report_id,
            "doctor_name": diagnosis.doctor.name,
            "patient_name": diagnosis.patient.name,
            "patient_age": diagnosis.patient.age,
            "patient_gender": diagnosis.patient.gender,
            "findings": findings,
            "heatmap_url": f"/static/{report_id}_heatmap.png",
            "pdf_url": f"/download/{report_id}",
            "metadata": {
                "processing_time": diagnosis.processing_time,
                "image_quality_score": diagnosis.image_quality_score,
                "has_abnormalities": diagnosis.has_abnormalities,
                "analysis_date": diagnosis.analyzed_at.strftime("%Y-%m-%d %H:%M:%S") if diagnosis.analyzed_at else "Unknown"
            }
        }
        
        return templates.TemplateResponse("report.html", response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report viewing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to load report")

@app.get("/api/stats")
async def get_statistics(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        from database import get_diagnosis_stats
        stats = get_diagnosis_stats(db, days=30)
        return stats
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        return {"error": "Failed to retrieve statistics"}

@app.get("/api/model-info")
async def get_model_information():
    """Get model information"""
    return get_model_info()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=True)
