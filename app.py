# app.py - PRODUCTION ONLY MediScan FastAPI Application for Render.com
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
from starlette.requests import Request
from sqlalchemy.orm import Session
from datetime import datetime

import os
import uuid
import time
import logging
import signal
import psutil
import gc
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

# Timeout handler for Render.com
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Analysis timeout")

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

# Create templates directory and add the HTML files
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")

# Configuration - Optimized for Render.com
MAX_FILE_SIZE = 50 * 1024 * 1024  # Reduced to 50MB for Render.com
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
    logger.info("Starting MediScan AI Application v2.0.0 - PRODUCTION ONLY")
    
    # Check model availability - MANDATORY for production
    if not check_model_availability():
        logger.error("CRITICAL: Model file 'model.pth.tar' not found!")
        logger.error("PRODUCTION MODE requires the trained model file.")
        raise RuntimeError("PRODUCTION MODE: Model file not found. Please ensure 'model.pth.tar' is available.")
    
    model_info = get_model_info()
    if model_info['mode'] != 'production':
        logger.error("CRITICAL: System not in production mode!")
        raise RuntimeError("PRODUCTION MODE: System must be in production mode")
    
    logger.info("✅ Production model loaded successfully")
    logger.info(f"✅ Model supports Grad-CAM: {model_info['gradcam_support']}")
    
    # Check database health
    health = check_database_health()
    if health['status'] == 'healthy':
        logger.info("✅ Database connection healthy")
    else:
        logger.error(f"❌ Database connection failed: {health.get('error', 'Unknown error')}")

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Main upload form page - PRODUCTION ONLY"""
    try:
        model_info = get_model_info()
        if model_info['mode'] != 'production':
            raise RuntimeError("PRODUCTION MODE: Model not available")
            
        return templates.TemplateResponse("form.html", {
            "request": request,
            "model_info": model_info,
            "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
            "supported_formats": list(ALLOWED_EXTENSIONS)
        })
    except Exception as e:
        logger.error(f"Error loading home page: {e}")
        # Return error page instead of demo mode
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head><title>MediScan AI - Production Error</title></head>
        <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
            <h1>MediScan AI - Production Mode Error</h1>
            <div style="background: #fee2e2; border: 1px solid #f87171; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <p><strong>System Error:</strong> Production model not available.</p>
                <p>Error: {str(e)}</p>
                <p>Please ensure the trained model file 'model.pth.tar' is present and accessible.</p>
            </div>
            <p>Contact system administrator to resolve this issue.</p>
        </body>
        </html>
        """, status_code=503)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model_info = get_model_info()
        db_health = check_database_health()
        memory = psutil.virtual_memory()
        
        # PRODUCTION MODE must have model available
        if not model_info['model_available'] or model_info['mode'] != 'production':
            return {
                "status": "error",
                "error": "PRODUCTION MODE: Model not available",
                "model_mode": model_info.get('mode', 'error'),
                "model_available": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return {
            "status": "healthy",
            "model_mode": "production",
            "model_available": True,
            "gradcam_available": model_info['gradcam_support'],
            "database": db_health,
            "memory_usage": f"{memory.percent}%",
            "version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/system/status")
async def system_status():
    """System status endpoint for monitoring"""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "mode": "production",
            "memory": {
                "total": memory.total // 1024 // 1024,  # MB
                "available": memory.available // 1024 // 1024,  # MB
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total // 1024 // 1024,  # MB
                "free": disk.free // 1024 // 1024,  # MB
                "percent": (disk.used / disk.total) * 100
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

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
    Analyze uploaded chest X-ray image - PRODUCTION ONLY
    """
    start_time = time.time()
    report_id = str(uuid.uuid4())
    
    # Set analysis timeout (4 minutes for Render's 5-minute limit)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(240)  # 4 minutes
    
    try:
        logger.info(f"Starting PRODUCTION analysis for report {report_id}")
        
        # MANDATORY: Check model availability
        if not check_model_availability():
            raise HTTPException(
                status_code=503, 
                detail="PRODUCTION MODE: Trained model not available. System cannot perform analysis."
            )
        
        model_info = get_model_info()
        if model_info['mode'] != 'production':
            raise HTTPException(
                status_code=503,
                detail="PRODUCTION MODE: System not properly configured for production analysis."
            )
        
        # Memory cleanup before starting
        gc.collect()
        
        # Check available memory
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            logger.warning(f"High memory usage: {memory.percent}%")
            gc.collect()
            
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                raise HTTPException(
                    status_code=503, 
                    detail="Server temporarily overloaded. Please try again in a few minutes."
                )
        
        # Log activity
        log_activity(
            db, "analysis_started", 
            f"PRODUCTION X-ray analysis initiated for patient {patient_name}",
            user_type="doctor",
            user_identifier=doctor_name
        )
        
        # Validate input parameters
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if patient_age < 0 or patient_age > 150:
            raise HTTPException(status_code=400, detail="Invalid age: must be between 0 and 150")
        
        if patient_gender.lower() not in ['male', 'female', 'other']:
            raise HTTPException(status_code=400, detail="Invalid gender: must be 'male', 'female', or 'other'")
        
        # Read file content with size limit
        try:
            contents = await file.read()
            if len(contents) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File too large: {len(contents)//1024//1024}MB. Maximum allowed: {MAX_FILE_SIZE//1024//1024}MB"
                )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
        
        # Security validation
        validator = SecurityValidator()
        
        if not validator.validate_file_type(file.filename, file.content_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        if not validator.validate_file_size(contents):
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        is_valid, validation_message = validator.validate_image_content(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_message)
        
        logger.info(f"File validation passed for {file.filename} (PRODUCTION mode)")
        
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
            'analysis_status': 'processing',
            'model_version': 'CheXNet-DenseNet121 (Production)'
        }
        
        # Get image dimensions
        try:
            image = Image.open(io.BytesIO(contents))
            diagnosis_data['image_dimensions'] = f"{image.size[0]}x{image.size[1]}"
        except:
            pass
        
        diagnosis = create_diagnosis(db, diagnosis_data)
        logger.info(f"Created diagnosis record: {diagnosis.id}")
        
        # Perform PRODUCTION AI analysis
        logger.info("Starting PRODUCTION AI analysis with Grad-CAM...")
        
        memory_before = psutil.virtual_memory().percent
        logger.info(f"Memory usage before analysis: {memory_before}%")
        
        # MANDATORY: Generate heatmap
        findings, heatmap_bytes, metadata = analyze_xray(
            contents, 
            threshold=0.4, 
            generate_heatmap=True  # MANDATORY
        )
        
        # CRITICAL CHECK: Ensure heatmap was generated
        if heatmap_bytes is None:
            logger.error("CRITICAL: Heatmap generation failed!")
            raise RuntimeError("PRODUCTION MODE: Grad-CAM heatmap generation failed")
        
        # Clean up memory after analysis
        del contents
        gc.collect()
        
        memory_after = psutil.virtual_memory().percent
        logger.info(f"Memory usage after analysis: {memory_after}%")
        
        analysis_time = time.time() - start_time
        logger.info(f"PRODUCTION analysis completed in {analysis_time:.2f}s")
        
        # Clear the alarm
        signal.alarm(0)
        
        # Update diagnosis with results
        diagnosis.analysis_status = 'completed'
        diagnosis.processing_time = analysis_time
        diagnosis.has_abnormalities = metadata.get('has_abnormalities', False)
        diagnosis.max_confidence = metadata.get('max_confidence', 0.0)
        diagnosis.analyzed_at = datetime.utcnow()
        diagnosis.image_quality_score = metadata.get('image_quality_score', 0.0)
        
        # Determine primary finding
        if metadata.get('has_abnormalities'):
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
                        'gradcam_generated': True  # Always true in production
                    }
                    findings_data.append(finding_data)
                except:
                    continue
        
        if findings_data:
            add_findings(db, diagnosis.id, findings_data)
        
        # Save heatmap image
        heatmap_filename = f"{report_id}_heatmap.png"
        heatmap_path = f"static/{heatmap_filename}"
        
        try:
            os.makedirs("static", exist_ok=True)
            with open(heatmap_path, "wb") as f:
                f.write(heatmap_bytes)
            logger.info(f"Grad-CAM heatmap saved: {heatmap_path}")
        except Exception as e:
            logger.error(f"Failed to save heatmap: {e}")
            raise RuntimeError("PRODUCTION MODE: Failed to save Grad-CAM heatmap")
        
        # Generate PDF report in background
        pdf_path = f"reports/{report_id}.pdf"
        os.makedirs("reports", exist_ok=True)
        
        background_tasks.add_task(
            generate_and_send_report,
            doctor_name, patient_name, patient_age, patient_gender,
            findings, heatmap_bytes, pdf_path, email, report_id, db
        )
        
        # Record metrics
        record_metric(db, "analysis_time", analysis_time, "performance", diagnosis_id=diagnosis.id)
        record_metric(db, "image_quality", metadata.get('image_quality_score', 0), "quality", diagnosis_id=diagnosis.id)
        record_metric(db, "findings_count", len(findings_data), "analysis", diagnosis_id=diagnosis.id)
        record_metric(db, "gradcam_generated", 1, "feature", diagnosis_id=diagnosis.id)
        
        # Log completion
        log_activity(
            db, "analysis_completed",
            f"PRODUCTION X-ray analysis completed for {patient_name}. Found {len(findings_data)} findings. Grad-CAM generated successfully.",
            user_type="doctor",
            user_identifier=doctor_name,
            diagnosis_id=diagnosis.id
        )
        
        # Memory cleanup before response
        gc.collect()
        
        # Prepare response data
        response_data = {
            "request": request,
            "report_id": report_id,
            "doctor_name": doctor_name,
            "patient_name": patient_name,
            "patient_age": patient_age,
            "patient_gender": patient_gender,
            "findings": findings,
            "heatmap_url": f"/static/{report_id}_heatmap.png",
            "pdf_url": f"/download/{report_id}",
            "metadata": metadata,
            "processing_time": f"{analysis_time:.2f}",
            "abnormalities_detected": metadata.get('has_abnormalities', False),
            "confidence_score": f"{metadata.get('max_confidence', 0) * 100:.1f}%",
            "image_quality": f"{metadata.get('image_quality_score', 0) * 100:.1f}%",
            "current_date": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "current_datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "analysis_date": datetime.now().strftime('%Y-%m-%d'),
            "is_demo_mode": False,  # Always False in production
            "model_mode": "production",
            "gradcam_generated": True
        }
        
        # Return results
        try:
            return templates.TemplateResponse("report.html", response_data)
        except Exception as template_error:
            logger.error(f"Template error: {template_error}")
            # Simple HTML fallback
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MediScan AI - Analysis Results</title>
                <meta charset="UTF-8">
            </head>
            <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
                <h1>MediScan AI - PRODUCTION Analysis Results</h1>
                <h2>Patient: {patient_name}</h2>
                <p><strong>Doctor:</strong> {doctor_name}</p>
                <p><strong>Processing Time:</strong> {analysis_time:.2f} seconds</p>
                <p><strong>Report ID:</strong> {report_id}</p>
                <p><strong>Mode:</strong> Production with Grad-CAM</p>
                <h3>Findings:</h3>
                <ul>
                {"".join(f"<li>{finding}</li>" for finding in findings)}
                </ul>
                <div style="margin: 20px 0;">
                    <h3>AI Grad-CAM Heatmap:</h3>
                    <img src="/static/{report_id}_heatmap.png" alt="Grad-CAM Heatmap" style="max-width: 400px; border: 1px solid #ccc;">
                </div>
                <p><a href="/download/{report_id}" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Download PDF Report</a></p>
                <p><a href="/">Analyze Another Image</a></p>
            </body>
            </html>
            """)
        
    except TimeoutException:
        logger.error(f"Analysis timeout for report {report_id}")
        signal.alarm(0)
        
        try:
            diagnosis = get_diagnosis_by_report_id(db, report_id)
            if diagnosis:
                diagnosis.analysis_status = 'timeout'
                db.commit()
        except:
            pass
        
        raise HTTPException(status_code=504, detail="Analysis timeout. Please try again with a smaller image or contact support.")
        
    except HTTPException:
        signal.alarm(0)
        raise
    except Exception as e:
        signal.alarm(0)
        logger.error(f"PRODUCTION analysis failed for report {report_id}: {str(e)}")
        
        try:
            diagnosis = get_diagnosis_by_report_id(db, report_id)
            if diagnosis:
                diagnosis.analysis_status = 'failed'
                db.commit()
        except:
            pass
        
        log_activity(
            db, "analysis_failed",
            f"PRODUCTION X-ray analysis failed: {str(e)}",
            user_type="doctor",
            user_identifier=doctor_name,
            success=False,
            error_message=str(e)
        )
        
        if "502" in str(e) or "Bad Gateway" in str(e):
            raise HTTPException(status_code=502, detail="Server temporarily unavailable. Please try again in a few minutes.")
        else:
            raise HTTPException(status_code=500, detail=f"PRODUCTION analysis failed: {str(e)}")

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
    """Background task to generate and send PDF report with Grad-CAM"""
    try:
        logger.info(f"Generating PRODUCTION PDF report with Grad-CAM for {report_id}")
        
        # Generate PDF with heatmap
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
        
        logger.info(f"PDF with Grad-CAM generated successfully: {pdf_path}")
        
        # Send email
        try:
            send_email(email, pdf_path, patient_name, doctor_name, report_id)
            
            if diagnosis:
                diagnosis.email_sent = True
                diagnosis.email_sent_at = datetime.utcnow()
                db.commit()
            
            logger.info(f"Email sent successfully to {email}")
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
        
        log_activity(
            db, "report_generated",
            f"PRODUCTION PDF report with Grad-CAM generated for {patient_name}",
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

@app.get("/api/model-info")
async def get_model_information():
    """Get model information"""
    return get_model_info()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=True)
