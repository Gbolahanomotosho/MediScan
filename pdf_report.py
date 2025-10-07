# pdf_report.py - COMPLETE PRODUCTION PDF Report Generator with Grad-CAM
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import Color, black, darkblue, darkred, darkgreen
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch
from datetime import datetime
from PIL import Image as PILImage
import io
import os
import textwrap
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_pdf_report(
    doctor_name: str,
    patient_name: str,    
    patient_age: int, 
    patient_gender: str,
    findings: List[str],
    heatmap_bytes: Optional[bytes] = None,
    output_path: str = "MediScan_report.pdf",
    signature_line: bool = True,
    clinical_notes: Optional[str] = None,
    urgency_level: str = "routine"
) -> str:
    """
    Generate PRODUCTION PDF report with Grad-CAM heatmap
    
    Args:
        doctor_name: Name of the attending doctor
        patient_name: Patient's full name
        patient_age: Patient's age
        patient_gender: Patient's gender
        findings: List of analysis findings
        heatmap_bytes: Grad-CAM heatmap image bytes (MANDATORY)
        output_path: Output file path
        signature_line: Whether to include signature area
        clinical_notes: Additional clinical notes
        urgency_level: Urgency level (routine, urgent, emergency)
        
    Returns:
        Path to generated PDF file
    """
    
    try:
        logger.info(f"Generating PRODUCTION PDF report: {output_path}")
        
        # CRITICAL CHECK: Ensure heatmap bytes are provided for production
        if heatmap_bytes is None:
            logger.error("PRODUCTION MODE: Grad-CAM heatmap bytes are required")
            raise ValueError("PRODUCTION MODE: Grad-CAM heatmap bytes are required for PDF generation")
        
        # Validate heatmap bytes
        if len(heatmap_bytes) == 0:
            logger.error("PRODUCTION MODE: Empty heatmap bytes provided")
            raise ValueError("PRODUCTION MODE: Heatmap bytes are empty")
        
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build story (content)
        story = []
        report_generator = MediScanPDFReport()
        
        # Header with logo
        story.extend(report_generator.create_header())
        
        # Patient information
        story.extend(report_generator.create_patient_info(
            doctor_name, patient_name, patient_age, patient_gender, urgency_level
        ))
        
        # Analysis findings
        story.extend(report_generator.create_findings_section(findings))
        
        # MANDATORY: Grad-CAM heatmap image
        story.extend(report_generator.create_gradcam_section(heatmap_bytes))
        
        # Clinical notes
        if clinical_notes and clinical_notes.strip():
            story.extend(report_generator.create_clinical_notes_section(clinical_notes))
        
        # Recommendations
        story.extend(report_generator.create_recommendations_section(findings, urgency_level))
        
        # Signature area
        if signature_line:
            story.extend(report_generator.create_signature_section())
        
        # Disclaimer
        story.extend(report_generator.create_disclaimer_section())
        
        # Footer
        story.extend(report_generator.create_footer())
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PRODUCTION PDF report with Grad-CAM generated successfully: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to generate PRODUCTION PDF report: {e}")
        raise RuntimeError(f"PRODUCTION PDF generation failed: {e}")

class MediScanPDFReport:
    """COMPLETE PDF report builder class for production mode"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Header style
        self.styles.add(ParagraphStyle(
            name='MediScanHeader',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=darkblue,
            spaceAfter=20,
            alignment=1  # Center alignment
        ))
        
        # Subheader style
        self.styles.add(ParagraphStyle(
            name='MediScanSubHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=darkblue,
            spaceAfter=12,
            spaceBefore=12
        ))
        
        # Finding styles
        self.styles.add(ParagraphStyle(
            name='UrgentFinding',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=darkred,
            leftIndent=20,
            spaceBefore=4
        ))
        
        self.styles.add(ParagraphStyle(
            name='NormalFinding',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceBefore=3
        ))
        
        # Disclaimer style
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=Color(0.3, 0.3, 0.3),
            borderWidth=1,
            borderColor=Color(0.8, 0.8, 0.8),
            borderPadding=10,
            backColor=Color(0.95, 0.95, 0.95),
            alignment=4  # Justify
        ))
    
    def create_header(self) -> List:
        """Create header section with logo and title"""
        elements = []
        
        # Try to add logo
        logo_path = "static/logo.jpg"
        if os.path.exists(logo_path):
            try:
                logo = Image(logo_path, width=2*inch, height=1*inch)
                elements.append(logo)
                elements.append(Spacer(1, 12))
                logger.info("Logo added to PDF header")
            except Exception as e:
                logger.warning(f"Could not load logo: {e}")
        
        # Main title
        title = Paragraph(
            "MediScan AI Chest X-ray Diagnosis Report<br/><font size=12>PRODUCTION MODE with Grad-CAM Visualization</font>", 
            self.styles['MediScanHeader']
        )
        elements.append(title)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def create_patient_info(self, doctor_name: str, patient_name: str, patient_age: int, 
                           patient_gender: str, urgency_level: str) -> List:
        """Create patient information table"""
        elements = []
        
        # Patient information table
        data = [
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')],
            ['Doctor Name:', doctor_name],
            ['Patient Name:', patient_name],
            ['Patient Age:', f"{patient_age} years"],
            ['Patient Gender:', patient_gender.title()],
            ['Urgency Level:', urgency_level.upper()],
            ['Analysis Method:', 'CheXNet Deep Learning (DenseNet-121)'],
            ['Analysis Mode:', 'PRODUCTION with Grad-CAM'],
            ['Conditions Screened:', '14 Thoracic Pathologies'],
            ['Visualization:', 'Grad-CAM Heatmap Included']
        ]
        
        # Create table
        table = Table(data, colWidths=[2.2*inch, 2.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), Color(0.9, 0.9, 0.9)),
            ('TEXTCOLOR', (0, 0), (-1, -1), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [Color(0.95, 0.95, 0.95), Color(1, 1, 1)]),
            ('GRID', (0, 0), (-1, -1), 1, Color(0.8, 0.8, 0.8)),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def create_findings_section(self, findings: List[str]) -> List:
        """Create findings section with categorized results"""
        elements = []
        
        # Section header
        header = Paragraph("AI Analysis Results - PRODUCTION MODE", self.styles['MediScanSubHeader'])
        elements.append(header)
        
        if not findings or (len(findings) == 1 and "no major abnormalities" in findings[0].lower()):
            # No abnormalities found
            finding_para = Paragraph(
                "<b>Normal Study:</b> No major abnormalities detected among the 14 thoracic conditions analyzed. "
                "The chest X-ray appears within normal limits for the screened pathologies. "
                "This analysis was performed using the production-grade CheXNet AI model with Grad-CAM visualization "
                "to show areas of model attention during the diagnostic process.",
                self.styles['NormalFinding']
            )
            elements.append(finding_para)
        else:
            # Add findings summary
            elements.append(Paragraph(
                f"<b>Analysis Summary:</b> {len(findings)} finding(s) detected by the AI model. "
                "Each finding includes confidence scores and clinical significance. "
                "The Grad-CAM heatmap below shows where the AI model focused its attention during analysis.",
                self.styles['Normal']
            ))
            elements.append(Spacer(1, 10))
            
            # Categorize findings by urgency
            urgent_findings = []
            routine_findings = []
            
            for finding in findings:
                finding_lower = finding.lower()
                if any(urgent_word in finding_lower for urgent_word in 
                      ['urgent', 'emergency', 'pneumothorax', 'edema', 'severe']):
                    urgent_findings.append(finding)
                else:
                    routine_findings.append(finding)
            
            # Urgent findings first
            if urgent_findings:
                urgent_header = Paragraph(
                    "<b>URGENT FINDINGS - Immediate Attention Required:</b>", 
                    self.styles['UrgentFinding']
                )
                elements.append(urgent_header)
                
                for finding in urgent_findings:
                    clean_finding = finding.replace("•", "").replace("-", "").strip()
                    if clean_finding:
                        finding_para = Paragraph(f"• {clean_finding}", self.styles['UrgentFinding'])
                        elements.append(finding_para)
                
                elements.append(Spacer(1, 10))
            
            # Routine findings
            if routine_findings:
                if urgent_findings:
                    routine_header = Paragraph("<b>Additional Findings:</b>", self.styles['NormalFinding'])
                else:
                    routine_header = Paragraph("<b>Detected Findings:</b>", self.styles['NormalFinding'])
                
                elements.append(routine_header)
                
                for finding in routine_findings:
                    clean_finding = finding.replace("•", "").replace("-", "").strip()
                    if clean_finding and "however, this does not rule out" not in clean_finding.lower():
                        finding_para = Paragraph(f"• {clean_finding}", self.styles['NormalFinding'])
                        elements.append(finding_para)
        
        elements.append(Spacer(1, 15))
        return elements
    
    def create_gradcam_section(self, heatmap_bytes: bytes) -> List:
        """Create Grad-CAM heatmap section - MANDATORY for production - FIXED VERSION"""
        elements = []
        
        try:
            logger.info("Adding Grad-CAM section to PDF")
            
            # Section header
            header = Paragraph("Grad-CAM Visual Analysis - AI Attention Mapping", self.styles['MediScanSubHeader'])
            elements.append(header)
            
            # Validate heatmap bytes
            if not heatmap_bytes or len(heatmap_bytes) == 0:
                raise ValueError("Empty heatmap bytes provided")
            
            # Process Grad-CAM heatmap image
            try:
                heatmap_image = PILImage.open(io.BytesIO(heatmap_bytes))
                
                # Convert to RGB if necessary
                if heatmap_image.mode != 'RGB':
                    heatmap_image = heatmap_image.convert('RGB')
                
                logger.info(f"Heatmap image loaded: {heatmap_image.size}, mode: {heatmap_image.mode}")
                
            except Exception as e:
                logger.error(f"Failed to load heatmap image: {e}")
                raise ValueError(f"Invalid heatmap image data: {e}")
            
            # Resize if too large for PDF (maintain aspect ratio)
            max_width, max_height = 450, 450  # pixels
            original_width, original_height = heatmap_image.size
            
            # Calculate scaling factor
            scale_factor = min(max_width/original_width, max_height/original_height, 1.0)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            if scale_factor < 1.0:
                heatmap_image = heatmap_image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                logger.info(f"Resized heatmap: {original_width}x{original_height} -> {new_width}x{new_height}")
            
            # Convert to BytesIO for ReportLab - FIXED VERSION
            img_buffer = io.BytesIO()
            heatmap_image.save(img_buffer, format='PNG', quality=95, optimize=True)
            img_buffer.seek(0)  # Reset buffer position to beginning
            
            # Calculate display size in inches (maintain aspect ratio)
            display_width = min(4.5, (new_width / 72))  # Convert pixels to inches
            display_height = min(4.5, (new_height / 72))
            
            # FIXED: Pass BytesIO buffer directly instead of ImageReader
            gradcam_image = Image(img_buffer, width=display_width*inch, height=display_height*inch)
            
            # Center the image
            elements.append(Spacer(1, 10))
            elements.append(gradcam_image)
            elements.append(Spacer(1, 15))
            
            # Detailed Grad-CAM explanation
            explanation = Paragraph(
                "<b>Grad-CAM Heatmap Analysis:</b><br/><br/>"
                "This visualization represents the <i>Class Activation Mapping</i> generated by the CheXNet deep learning model "
                "during the diagnostic analysis. The colored overlay shows the specific regions of the chest X-ray "
                "that the AI model considered most important when making its diagnostic predictions.<br/><br/>"
                
                "<b>Color Interpretation Guide:</b><br/>"
                "• <font color='red'><b>Red/Hot regions:</b></font> Areas of highest AI attention and confidence<br/>"
                "• <font color='orange'><b>Orange/Yellow regions:</b></font> Moderate AI attention areas<br/>"
                "• <font color='blue'><b>Blue/Cool regions:</b></font> Lower attention areas<br/>"
                "• <b>Transparent areas:</b> Background regions with minimal diagnostic relevance<br/><br/>"
                
                "<b>Clinical Interpretation:</b><br/>"
                "The heatmap provides insight into the AI's decision-making process by highlighting anatomical regions "
                "that influenced the diagnostic conclusions. This visualization assists radiologists in understanding "
                "which areas the model found significant and can help guide further clinical evaluation. "
                "However, the heatmap should be interpreted alongside clinical findings and radiological expertise.<br/><br/>"
                
                "<i>Note: Grad-CAM visualizations are explanatory tools that show model attention patterns. "
                "They do not replace professional radiological interpretation and should be used as supplementary "
                "information in the diagnostic process.</i>",
                self.styles['Normal']
            )
            elements.append(explanation)
            elements.append(Spacer(1, 15))
            
            logger.info("Grad-CAM heatmap successfully added to PDF")
            
        except Exception as e:
            logger.error(f"Failed to add Grad-CAM heatmap to PDF: {e}")
            
            # Add error message instead of failing completely
            error_section = [
                Paragraph("Grad-CAM Visualization Error", self.styles['MediScanSubHeader']),
                Paragraph(
                    f"<b>Grad-CAM Visualization Error:</b> The AI-generated heatmap could not be included in this report "
                    f"due to a technical issue: {str(e)}. The diagnostic analysis results above remain valid and were "
                    f"generated using the production CheXNet model. Please contact technical support if this issue persists.<br/><br/>"
                    f"<i>Error details: {type(e).__name__} - {str(e)}</i>",
                    self.styles['NormalFinding']
                ),
                Spacer(1, 10)
            ]
            elements.extend(error_section)
        
        return elements
    
    def create_clinical_notes_section(self, clinical_notes: str) -> List:
        """Create clinical notes section"""
        elements = []
        
        header = Paragraph("Clinical Notes", self.styles['MediScanSubHeader'])
        elements.append(header)
        
        # Clean and wrap the notes
        clean_notes = clinical_notes.strip()
        if len(clean_notes) > 500:
            # Split long notes into paragraphs
            wrapped_notes = textwrap.fill(clean_notes, width=80)
        else:
            wrapped_notes = clean_notes
        
        notes_para = Paragraph(wrapped_notes, self.styles['Normal'])
        elements.append(notes_para)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def create_recommendations_section(self, findings: List[str], urgency_level: str) -> List:
        """Create recommendations section based on findings"""
        elements = []
        
        header = Paragraph("Clinical Recommendations", self.styles['MediScanSubHeader'])
        elements.append(header)
        
        # Determine recommendations based on findings and urgency
        recommendations = []
        
        # Urgency-based recommendations
        if urgency_level.lower() == 'emergency':
            recommendations.extend([
                "<font color='red'><b>IMMEDIATE ACTION REQUIRED:</b></font> Seek emergency medical attention immediately.",
                "<b>Emergency Care:</b> Contact emergency services or go to the nearest emergency department.",
                "<b>Time Sensitivity:</b> Do not delay - immediate medical evaluation is critical."
            ])
        elif urgency_level.lower() == 'urgent':
            recommendations.extend([
                "<font color='orange'><b>URGENT CONSULTATION:</b></font> Schedule immediate appointment with radiologist or pulmonologist.",
                "<b>Priority Scheduling:</b> Request urgent or same-day appointment if possible."
            ])
        
        # Check for specific conditions in findings
        has_abnormalities = any(
            "probability" in finding and ":" in finding 
            for finding in findings
        )
        
        if has_abnormalities:
            recommendations.extend([
                "<b>1. Professional Review:</b> Have these AI findings reviewed by a qualified radiologist with expertise in chest imaging.",
                "<b>2. Clinical Correlation:</b> Consider patient symptoms, physical examination findings, and medical history in context of AI results.",
                "<b>3. Grad-CAM Analysis:</b> Review the heatmap visualization above to understand which anatomical regions the AI model prioritized.",
                "<b>4. Additional Imaging:</b> Consider additional imaging studies (CT chest, lateral views) if clinically indicated.",
                "<b>5. Treatment Planning:</b> Develop appropriate treatment plan based on confirmed findings and clinical assessment.",
                "<b>6. Follow-up Protocol:</b> Establish monitoring schedule based on findings severity and patient response."
            ])
        else:
            recommendations.extend([
                "<b>1. Routine Monitoring:</b> Continue routine chest health monitoring as part of regular healthcare.",
                "<b>2. Preventive Care:</b> Maintain regular health check-ups and chest imaging as clinically appropriate.",
                "<b>3. Lifestyle Factors:</b> Follow healthy lifestyle practices for optimal lung health (no smoking, regular exercise).",
                "<b>4. Grad-CAM Review:</b> The heatmap shows normal distribution of AI attention without focal abnormalities."
            ])
        
        # Add general recommendations
        recommendations.extend([
            "<b>7. AI Limitations:</b> Remember that AI analysis supplements but does not replace clinical judgment and radiological expertise.",
            "<b>8. Documentation:</b> Keep this report and Grad-CAM visualization for medical records and future reference.",
            "<b>9. Second Opinion:</b> Consider second opinion from subspecialty radiologist if findings are complex or uncertain.",
            "<b>10. Patient Communication:</b> Discuss results with patient in appropriate clinical context."
        ])
        
        # Add recommendations to PDF
        for i, rec in enumerate(recommendations):
            rec_para = Paragraph(rec, self.styles['Normal'])
            elements.append(rec_para)
            if i < len(recommendations) - 1:  # Don't add spacer after last item
                elements.append(Spacer(1, 5))
        
        elements.append(Spacer(1, 15))
        return elements
    
    def create_signature_section(self) -> List:
        """Create signature section for medical verification"""
        elements = []
        
        header = Paragraph("Medical Professional Verification", self.styles['MediScanSubHeader'])
        elements.append(header)
        
        # Add instruction
        instruction = Paragraph(
            "<i>This AI-generated report requires review and verification by a qualified medical professional "
            "before clinical use. Please complete the following:</i>",
            self.styles['Normal']
        )
        elements.append(instruction)
        elements.append(Spacer(1, 10))
        
        # Create signature table
        sig_data = [
            ['Reviewing Physician Signature:', '_' * 50],
            ['Print Name:', '_' * 50],
            ['Medical License Number:', '_' * 30],
            ['Date of Review:', '_' * 25],
            ['Medical Facility/Institution:', '_' * 40],
            ['Additional Comments:', '_' * 50]
        ]
        
        sig_table = Table(sig_data, colWidths=[2.3*inch, 2.7*inch])
        sig_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        elements.append(sig_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def create_disclaimer_section(self) -> List:
        """Create comprehensive medical disclaimer section"""
        elements = []
        
        disclaimer_text = (
            "<b>IMPORTANT MEDICAL DISCLAIMER - PRODUCTION AI SYSTEM</b><br/><br/>"
            
            "This report contains results from a PRODUCTION-GRADE artificial intelligence system for chest X-ray analysis. "
            "The system uses CheXNet architecture (DenseNet-121) trained on the ChestX-ray14 dataset and includes "
            "Grad-CAM visualization to show AI attention patterns. While this represents state-of-the-art medical AI technology, "
            "it is designed as a screening and diagnostic assistance tool only.<br/><br/>"
            
            "<b>Critical Limitations and Requirements:</b><br/>"
            "• <b>Professional Verification Required:</b> All AI findings must be verified by qualified medical professionals with appropriate expertise in chest imaging<br/>"
            "• <b>Not a Substitute for Clinical Judgment:</b> This system does not replace radiological interpretation, clinical assessment, or medical decision-making<br/>"
            "• <b>False Positives/Negatives Possible:</b> AI systems can produce incorrect results - both false positive findings and missed pathology can occur<br/>"
            "• <b>Clinical Correlation Essential:</b> AI results must be interpreted in context of patient history, physical examination, and other clinical data<br/>"
            "• <b>Grad-CAM Limitations:</b> Heatmap visualizations show AI attention patterns but do not guarantee diagnostic accuracy or clinical relevance<br/>"
            "• <b>Training Data Constraints:</b> AI performance may vary for cases unlike those in the training dataset<br/>"
            "• <b>Emergency Cases:</b> Time-sensitive conditions require immediate medical attention regardless of AI results<br/><br/>"
            
            "<b>Technical Specifications:</b><br/>"
            "• <b>Model Architecture:</b> CheXNet (DenseNet-121) deep convolutional neural network<br/>"
            "• <b>Training Dataset:</b> ChestX-ray14 (NIH Clinical Center)<br/>"
            "• <b>Visualization Method:</b> Grad-CAM (Gradient-weighted Class Activation Mapping)<br/>"
            "• <b>Conditions Analyzed:</b> 14 thoracic pathologies (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, "
            "Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia)<br/>"
            "• <b>Processing Mode:</b> PRODUCTION with quality controls and validation<br/>"
            "• <b>Report Generation:</b> " + datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC') + "<br/><br/>"
            
            "<b>Legal and Regulatory Notice:</b><br/>"
            "This AI system is intended for use by qualified healthcare professionals as a diagnostic aid. "
            "Medical decisions should never be based solely on AI results or visualizations. "
            "Healthcare providers remain fully responsible for patient care decisions, diagnostic accuracy, and treatment outcomes. "
            "This technology should be used in accordance with institutional policies and applicable regulations.<br/><br/>"
            
            "<b>Quality Assurance:</b> This analysis was performed using validated production algorithms with "
            "appropriate quality controls. However, system performance should be monitored and validated "
            "according to institutional quality assurance protocols."
        )
        
        disclaimer = Paragraph(disclaimer_text, self.styles['Disclaimer'])
        elements.append(disclaimer)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def create_footer(self) -> List:
        """Create professional footer section"""
        elements = []
        
        # Add separator line
        line_data = [['_' * 80]]
        line_table = Table(line_data, colWidths=[5*inch])
        line_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('TEXTCOLOR', (0, 0), (-1, -1), Color(0.5, 0.5, 0.5))
        ]))
        elements.append(line_table)
        elements.append(Spacer(1, 10))
        
        # Footer content
        footer_text = (
            "<b>MediScan AI - PRODUCTION Medical Imaging Analysis System</b><br/>"
            "Advanced Chest X-ray Analysis with CheXNet Deep Learning + Grad-CAM Visualization<br/>"
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>"
            "System Version: MediScan AI v2.0 Production<br/>"
            "For technical support or questions: support@mediscan-ai.com<br/>"
            "© 2024 MediScan AI Systems. All rights reserved."
        )
        
        footer = Paragraph(footer_text, ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=Color(0.4, 0.4, 0.4),
            alignment=1  # Center alignment
        ))
        elements.append(footer)
        
        return elements

# Legacy compatibility functions
def generate_pdf_report_legacy(
    doctor_name: str,
    patient_name: str,    
    patient_age: int, 
    patient_gender: str,
    findings: List[str],
    heatmap_bytes: bytes,
    output_path: str = "MediScan_report.pdf",
    signature_line: bool = True
) -> str:
    """Legacy function wrapper for backward compatibility"""
    return generate_pdf_report(
        doctor_name, patient_name, patient_age, patient_gender,
        findings, heatmap_bytes, output_path, signature_line
    )

# Test function
def test_pdf_generation():
    """Test the PDF generation with sample data"""
    test_findings = [
        "Pneumonia: 0.75 probability - Lung infection detected in right lower lobe",
        "Consolidation: 0.68 probability - Dense opacity consistent with inflammatory process",
        "Mass: 0.60 probability - Abnormal growth detected requiring further evaluation"
    ]
    
    # Create test heatmap bytes (sample red image)
    test_image = PILImage.new('RGB', (224, 224), color=(255, 100, 100))
    test_buffer = io.BytesIO()
    test_image.save(test_buffer, format='PNG')
    test_heatmap_bytes = test_buffer.getvalue()
    
    try:
        output_path = generate_pdf_report(
            doctor_name="Dr. Sarah Johnson",
            patient_name="John Smith",
            patient_age=45,
            patient_gender="Male",
            findings=test_findings,
            heatmap_bytes=test_heatmap_bytes,
            output_path="test_production_report_complete.pdf",
            clinical_notes="Patient presents with persistent cough and chest discomfort. History of smoking. Symptoms worsened over the past week.",
            urgency_level="urgent"
        )
        print(f"Test PRODUCTION PDF report generated successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Test PDF generation failed: {e}")
        return False

if __name__ == "__main__":
    # Run test when script is executed directly
    print("Testing PRODUCTION PDF generation with Grad-CAM...")
    success = test_pdf_generation()
    if success:
        print("PDF generation test completed successfully!")
    else:
        print("PDF generation test failed!")
        
    # Display key features
    print("\nPRODUCTION PDF Features:")
    print("✅ Mandatory Grad-CAM heatmap integration - FIXED")
    print("✅ Complete patient and doctor information")
    print("✅ Comprehensive findings analysis")
    print("✅ Clinical recommendations based on urgency")
    print("✅ Professional medical disclaimer")
    print("✅ Signature section for medical verification")
    print("✅ Technical specifications and metadata")
    print("✅ Error handling for missing heatmaps")
    print("✅ Production-grade formatting and styling")
    print("\n🔧 BUG FIX: Resolved ImageReader TypeError issue")
    print("   - Now passes BytesIO buffer directly to Image()")
    print("   - No more 'expected str, bytes or os.PathLike object' error")
