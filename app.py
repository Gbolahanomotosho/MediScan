# app.py - Memory Optimized MediScan for Render.com
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
import asyncio
from datetime import datetime
from typing import Optional, List, Dict
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

# RENDER-SPECIFIC CONFIGURATION
MAX_REQUEST_TIME = 25  # Render has 30s timeout, leave 5s buffer
MEMORY_LIMIT_MB = 400  # Stay under 512MB
MAX_FILE_SIZE = 10 * 1024 * 1024  # Reduced to 10MB for faster processing

class RenderTimeoutHandler:
    """Handle Render.com specific timeouts and memory issues"""
    
    def __init__(self, max_time: int = MAX_REQUEST_TIME):
        self.max_time = max_time
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def check_timeout(self):
        if self.start_time and (time.time() - self.start_time) > self.max_time:
            raise HTTPException(
                status_code=504,
                detail="Request timeout to prevent server overload. Please try with a smaller image."
            )
    
    def get_remaining_time(self) -> float:
        if not self.start_time:
            return self.max_time
        return max(0, self.max_time - (time.time() - self.start_time))

# Initialize FastAPI app with optimized settings
app = FastAPI(
    title="MediScan AI - Advanced Chest X-ray Diagnosis System",
    description="AI-powered chest X-ray analysis optimized for cloud deployment",
    version="2.0.1"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}  # Removed TIFF for simplicity
MIN_IMAGE_SIZE = (224, 224)

# Initialize database
init_db()

class MemoryMonitor:
    """Monitor memory usage to prevent crashes"""
    
    @staticmethod
    def get_memory_percent() -> float:
        try:
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    @staticmethod
    def check_memory_safe(threshold: float = 80.0) -> bool:
        memory_percent = MemoryMonitor.get_memory_percent()
        if memory_percent > threshold:
            logger.warning(f"High memory usage: {memory_percent}%")
            gc.collect()  # Force garbage collection
        return memory_percent < 90.0  # Critical threshold
    
    @staticmethod
    def force_cleanup():
        """Force memory cleanup"""
        gc.collect()
        # Additional cleanup if needed

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup with memory monitoring"""
    logger.info("Starting MediScan AI Application v2.0.1 (Render.com Optimized)")
    
    # Check memory at startup
    memory_percent = MemoryMonitor.get_memory_percent()
    logger.info(f"Startup memory usage: {memory_percent}%")
    
    # Check model availability
    model_info = get_model_info()
    if model_info['mode'] == 'demo':
        logger.warning("⚠️  Running in DEMO MODE - No trained model available")
    else:
        logger.info("✅ Production model available")
    
    # Check database health
    health = check_database_health()
    if health['status'] == 'healthy':
        logger.info("✅ Database connection healthy")
    else:
        logger.error(f"❌ Database connection failed: {health.get('error', 'Unknown error')}")

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Main upload form page with memory awareness"""
    try:
        if not MemoryMonitor.check_memory_safe():
            logger.warning("High memory usage detected")
        
        model_info = get_model_info()
        return templates.TemplateResponse("form.html", {
            "request": request,
            "model_info": model_info,
            "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
            "supported_formats": list(ALLOWED_EXTENSIONS)
        })
    except Exception as e:
        logger.error(f"Error loading home page: {e}")
        # Return simplified HTML if template fails
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head><title>MediScan AI</title></head>
        <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
            <h1>MediScan AI - Chest X-ray Analysis</h1>
            <div style="background: #e3f2fd; border: 1px solid #2196f3; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <p><strong>System Status:</strong> Ready for analysis</p>
                <p>Upload a chest X-ray image for AI-powered analysis.</p>
            </div>
            <form action="/analyze/" method="post" enctype="multipart/form-data">
                <div style="margin: 15px 0;">
                    <label>Patient Name: <input type="text" name="patient_name" required style="margin-left: 10px; padding: 5px;"></label>
                </div>
                <div style="margin: 15px 0;">
                    <label>Doctor Name: <input type="text" name="doctor_name" required style="margin-left: 10px; padding: 5px;"></label>
                </div>
                <div style="margin: 15px 0;">
                    <label>Patient Age: <input type="number" name="patient_age" required style="margin-left: 10px; padding: 5px;"></label>
                </div>
                <div style="margin: 15px 0;">
                    <label>Gender: 
                        <select name="patient_gender" required style="margin-left: 10px; padding: 5px;">
                            <option value="">Select</option>
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                            <option value="other">Other</option>
                        </select>
                    </label>
                </div>
                <div style="margin: 15px 0;">
                    <label>Email: <input type="email" name="email" required style="margin-left: 10px; padding: 5px;"></label>
                </div>
                <div style="margin: 15px 0;">
                    <label>X-ray Image (Max {MAX_FILE_SIZE//1024//1024}MB): <input type="file" name="file" accept="image/*" required style="margin-left: 10px; padding: 5px;"></label>
                </div>
                <button type="submit" style="background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
                    Start Analysis
                </button>
            </form>
        </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Enhanced health check with memory monitoring"""
    try:
        model_info = get_model_info()
        db_health = check_database_health()
        memory = psutil.virtual_memory()
        
        # Check if system is healthy
        is_healthy = (
            memory.percent < 90 and
            db_health['status'] == 'healthy'
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "model_mode": model_info['mode'],
            "model_available": model_info['model_available'],
            "database": db_health,
            "memory_usage": f"{memory.percent:.1f}%",
            "memory_available_mb": memory.available // 1024 // 1024,
            "version": "2.0.1",
            "optimized_for": "Render.com",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
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
    Memory and timeout optimized X-ray analysis for Render.com
    """
    timeout_handler = RenderTimeoutHandler()
    timeout_handler.start()
    report_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting analysis for report {report_id} (Render optimized)")
        
        # Early memory check
        if not MemoryMonitor.check_memory_safe(threshold=85):
            raise HTTPException(
                status_code=503,
                detail="Server memory usage too high. Please try again in a moment."
            )
        
        timeout_handler.check_timeout()
        
        # Get model info to determine mode
        model_info = get_model_info()
        is_demo_mode = model_info['mode'] == 'demo'
        
        # Log activity
        log_activity(
            db, "analysis_started", 
            f"X-ray analysis initiated for patient {patient_name} ({model_info['mode']} mode)",
            user_type="doctor",
            user_identifier=doctor_name
        )
        
        # Validate input parameters
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if patient_age < 0 or patient_age > 150:
            raise HTTPException(status_code=400, detail="Invalid age: must be between 0 and 150")
        
        if patient_gender.lower() not in ['male', 'female', 'other']:
            raise HTTPException(status_code=400, detail="Invalid gender")
        
        timeout_handler.check_timeout()
        
        # Read and validate file with size limits for Render
        try:
            contents = await file.read()
            if len(contents) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File too large: {len(contents)//1024//1024}MB. Maximum: {MAX_FILE_SIZE//1024//1024}MB"
                )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
        
        # Quick file type validation
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        timeout_handler.check_timeout()
        
        # Quick image validation
        try:
            image = Image.open(io.BytesIO(contents))
            width, height = image.size
            
            if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image too small: {width}x{height}. Minimum: {MIN_IMAGE_SIZE[0]}x{MIN_IMAGE_SIZE[1]}"
                )
            
            # Optimize large images immediately
            max_dimension = 1024
            if max(width, height) > max_dimension:
                ratio = max_dimension / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert back to bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=85, optimize=True)
                contents = img_buffer.getvalue()
                logger.info(f"Resized image to {new_size} to optimize processing")
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        timeout_handler.check_timeout()
        
        logger.info(f"File validation passed for {file.filename}")
        
        # Database operations
        patient = get_patient_by_email(db, email)
        if not patient:
            patient_data = {
                'name': patient_name,
                'age': patient_age,
                'gender': patient_gender,
                'email': email
            }
            patient = create_patient(db, patient_data)
        
        doctor = get_doctor_by_name(db, doctor_name)
        if not doctor:
            doctor_data = {
                'name': doctor_name,
                'specialization': 'Radiologist'
            }
            doctor = create_doctor(db, doctor_data)
        
        timeout_handler.check_timeout()
        
        # Create diagnosis record
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
            'model_version': f"{model_info.get('model_architecture', 'Unknown')} ({'Demo' if is_demo_mode else 'Production'})"
        }
        
        diagnosis = create_diagnosis(db, diagnosis_data)
        logger.info(f"Created diagnosis record: {diagnosis.id}")
        
        timeout_handler.check_timeout()
        
        # Memory cleanup before analysis
        MemoryMonitor.force_cleanup()
        
        # Perform AI analysis with remaining time
        remaining_time = timeout_handler.get_remaining_time()
        logger.info(f"Starting analysis with {remaining_time:.1f}s remaining")
        
        try:
            findings, heatmap_bytes, metadata = analyze_xray(
                contents, 
                threshold=0.4, 
                generate_heatmap=True
            )
        except Exception as analysis_error:
            logger.error(f"Analysis error: {analysis_error}")
            # Fallback to demo mode if analysis fails
            if not is_demo_mode:
                logger.info("Falling back to demo mode due to analysis failure")
                findings, heatmap_bytes, metadata = analyze_xray(
                    contents, threshold=0.4, generate_heatmap=True
                )
                is_demo_mode = True
            else:
                raise analysis_error
        
        timeout_handler.check_timeout()
        
        # Process results
        analysis_time = metadata.get('processing_time', time.time() - timeout_handler.start_time)
        logger.info(f"Analysis completed in {analysis_time:.2f}s")
        
        # Update diagnosis with results
        diagnosis.analysis_status = 'completed'
        diagnosis.processing_time = analysis_time
        diagnosis.has_abnormalities = metadata.get('has_abnormalities', False)
        diagnosis.max_confidence = metadata.get('max_confidence', 0.0)
        diagnosis.analyzed_at = datetime.utcnow()
        diagnosis.image_quality_score = metadata.get('image_quality_score', 0.7)
        
        # Count findings
        finding_count = len([f for f in findings if "probability" in f])
        diagnosis.abnormalities_count = finding_count
        
        # Determine primary finding
        if metadata.get('has_abnormalities') and finding_count > 0:
            # Simple extraction of primary finding
            for finding in findings:
                if ":" in finding and "probability" in finding:
                    diagnosis.primary_finding = finding.split(":")[0].replace("•", "").strip()
                    break
        
        db.commit()
        
        timeout_handler.check_timeout()
        
        # Save heatmap
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
        
        # Generate PDF in background (non-blocking)
        pdf_path = f"reports/{report_id}.pdf"
        background_tasks.add_task(
            generate_and_send_report,
            doctor_name, patient_name, patient_age, patient_gender,
            findings, heatmap_bytes, pdf_path, email, report_id, db, is_demo_mode
        )
        
        # Record metrics
        record_metric(db, "analysis_time", analysis_time, "performance", diagnosis_id=diagnosis.id)
        record_metric(db, "memory_usage", MemoryMonitor.get_memory_percent(), "system", diagnosis_id=diagnosis.id)
        record_metric(db, "findings_count", finding_count, "analysis", diagnosis_id=diagnosis.id)
        
        # Log completion
        log_activity(
            db, "analysis_completed",
            f"Analysis completed for {patient_name}. Found {finding_count} findings. Mode: {model_info['mode']}",
            user_type="doctor",
            user_identifier=doctor_name,
            diagnosis_id=diagnosis.id
        )
        
        # Final memory cleanup
        MemoryMonitor.force_cleanup()
        
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
            "image_quality": f"{metadata.get('image_quality_score', 0) * 100:.1f}%",
            "current_date": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "current_datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "analysis_date": datetime.now().strftime('%Y-%m-%d'),
            "is_demo_mode": is_demo_mode,
            "model_mode": model_info['mode'],
            "render_optimized": True
        }
        
        # Return results with error handling
        try:
            return templates.TemplateResponse("report.html", response_data)
        except Exception as template_error:
            logger.error(f"Template error: {template_error}")
            # Simplified HTML fallback for Render
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MediScan AI - Analysis Results</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; padding: 20px; }}
                    .demo-banner {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                    .success {{ background: #d4edda; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                    .findings {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>MediScan AI - Analysis Results</h1>
                {'<div class="demo-banner"><strong>Demo Mode:</strong> Sample results for demonstration.</div>' if is_demo_mode else '<div class="success"><strong>Analysis Complete:</strong> Professional AI analysis completed.</div>'}
                
                <h2>Patient Information</h2>
                <p><strong>Patient:</strong> {patient_name}</p>
                <p><strong>Doctor:</strong> {doctor_name}</p>
                <p><strong>Processing Time:</strong> {analysis_time:.2f} seconds</p>
                <p><strong>Report ID:</strong> {report_id}</p>
                
                <h2>Analysis Results</h2>
                <div class="findings">
                    <h3>Findings:</h3>
                    <ul>
                        {"".join(f"<li>{finding}</li>" for finding in findings)}
                    </ul>
                </div>
                
                {'<div style="margin: 20px 0;"><h3>Analysis Visualization</h3><img src="' + f'/static/{report_id}_heatmap.png' + '" alt="Analysis Heatmap" style="max-width: 400px; border: 1px solid #ccc; border-radius: 5px;"></div>' if heatmap_path else ''}
                
                <div style="margin: 30px 0;">
                    <h3>Actions</h3>
                    <p><a href="/download/{report_id}" style="background: #007bff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 10px 10px 0;">📄 Download PDF Report</a></p>
                    <p><a href="/" style="background: #28a745; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">🔍 Analyze Another Image</a></p>
                </div>
                
                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #007bff;">
                    <p><strong>Note:</strong> This analysis has been optimized for cloud deployment. A detailed PDF report has been generated and will be emailed to you.</p>
                    <p><strong>Medical Disclaimer:</strong> This AI analysis is for screening purposes only. Always consult qualified healthcare professionals for medical advice.</p>
                </div>
            </body>
            </html>
            """)
        
    except HTTPException:
        # Let FastAPI handle HTTP exceptions normally
        raise
    except Exception as e:
        logger.error(f"Analysis failed for report {report_id}: {str(e)}")
        
        # Update diagnosis status if possible
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
        
        # Clean up memory on error
        MemoryMonitor.force_cleanup()
        
        # Return user-friendly error based on error type
        if "timeout" in str(e).lower() or "504" in str(e):
            raise HTTPException(
                status_code=504,
                detail="Analysis took too long. Please try with a smaller image file (under 5MB)."
            )
        elif "memory" in str(e).lower() or "503" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Server temporarily overloaded. Please wait a moment and try again."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Analysis failed. Please ensure you uploaded a valid X-ray image and try again."
            )

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
    db: Session,
    is_demo_mode: bool = False
):
    """Background task optimized for Render - non-blocking report generation"""
    try:
        logger.info(f"Generating PDF report for {report_id} ({'demo' if is_demo_mode else 'production'} mode)")
        
        # Add demo mode note to findings if applicable
        if is_demo_mode:
            demo_findings = ["[DEMO MODE] This is a sample analysis for demonstration purposes."] + findings
        else:
            demo_findings = findings
        
        # Generate PDF with timeout protection
        try:
            generate_pdf_report(
                doctor_name, patient_name, patient_age, patient_gender,
                demo_findings, heatmap_bytes, output_path=pdf_path
            )
            
            # Update diagnosis
            diagnosis = get_diagnosis_by_report_id(db, report_id)
            if diagnosis:
                diagnosis.pdf_generated = True
                diagnosis.pdf_path = pdf_path
                db.commit()
            
            logger.info(f"PDF generated successfully: {pdf_path}")
            
        except Exception as pdf_error:
            logger.error(f"PDF generation failed: {pdf_error}")
            # Continue without PDF - don't fail the whole process
        
        # Send email only in production mode
        if not is_demo_mode:
            try:
                send_email(email, pdf_path, patient_name, doctor_name, report_id)
                
                # Update diagnosis
                if diagnosis:
                    diagnosis.email_sent = True
                    diagnosis.email_sent_at = datetime.utcnow()
                    db.commit()
                
                logger.info(f"Email sent successfully to {email}")
                
            except Exception as e:
                logger.error(f"Email sending failed: {e}")
                # Don't fail - user can still download PDF
        else:
            logger.info("Demo mode: Skipping email send")
        
        # Log activity
        log_activity(
            db, "report_generated",
            f"{'Demo' if is_demo_mode else 'Production'} PDF report generated for {patient_name}",
            diagnosis_id=diagnosis.id if diagnosis else None
        )
        
        # Final cleanup
        MemoryMonitor.force_cleanup()
        
    except Exception as e:
        logger.error(f"Background report generation failed: {e}")
        # Don't raise - this is a background task

@app.get("/download/{report_id}")
async def download_pdf(report_id: str, db: Session = Depends(get_db)):
    """Download PDF report with proper error handling"""
    try:
        diagnosis = get_diagnosis_by_report_id(db, report_id)
        if not diagnosis:
            raise HTTPException(status_code=404, detail="Report not found")
        
        pdf_path = f"reports/{report_id}.pdf"
        if not os.path.exists(pdf_path):
            # Check if PDF is still being generated
            if diagnosis.analysis_status == 'completed' and not diagnosis.pdf_generated:
                raise HTTPException(
                    status_code=202,
                    detail="PDF report is being generated. Please try again in a moment."
                )
            else:
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

@app.get("/api/model-info")
async def get_model_information():
    """Get model information with memory status"""
    model_info = get_model_info()
    model_info.update({
        "memory_usage": f"{MemoryMonitor.get_memory_percent():.1f}%",
        "render_optimized": True,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "timeout_seconds": MAX_REQUEST_TIME
    })
    return model_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)  # Disable reload for production
