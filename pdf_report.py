# pdf_report.py - Enhanced PDF Report Generator
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

class MediScanPDFReport:
    """Advanced PDF report generator for MediScan AI"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
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
        
        # Finding style for urgent items
        self.styles.add(ParagraphStyle(
            name='UrgentFinding',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=darkred,
            leftIndent=20,
            bulletIndent=10
        ))
        
        # Finding style for normal items
        self.styles.add(ParagraphStyle(
            name='NormalFinding',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            bulletIndent=10
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
            backColor=Color(0.95, 0.95, 0.95)
        ))

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
    Generate comprehensive PDF report for MediScan AI analysis
    
    Args:
        doctor_name: Name of the attending doctor
        patient_name: Patient's full name
        patient_age: Patient's age
        patient_gender: Patient's gender
        findings: List of analysis findings
        heatmap_bytes: Heatmap image bytes
        output_path: Output file path
        signature_line: Whether to include signature area
        clinical_notes: Additional clinical notes
        urgency_level: Urgency level (routine, urgent, emergency)
        
    Returns:
        Path to generated PDF file
    """
    
    try:
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
        story.extend(report_generator._create_header())
        
        # Patient information
        story.extend(report_generator._create_patient_info(
            doctor_name, patient_name, patient_age, patient_gender, urgency_level
        ))
        
        # Analysis findings
        story.extend(report_generator._create_findings_section(findings))
        
        # Heatmap image
        if heatmap_bytes:
            story.extend(report_generator._create_image_section(heatmap_bytes))
        
        # Clinical notes
        if clinical_notes and clinical_notes.strip():
            story.extend(report_generator._create_clinical_notes_section(clinical_notes))
        
        # Recommendations
        story.extend(report_generator._create_recommendations_section(findings, urgency_level))
        
        # Signature area
        if signature_line:
            story.extend(report_generator._create_signature_section())
        
        # Disclaimer
        story.extend(report_generator._create_disclaimer_section())
        
        # Footer
        story.extend(report_generator._create_footer())
        
        # Build PDF
        doc.build(story)
        
        print(f"PDF report generated successfully: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Failed to generate PDF report: {e}")
        raise

class MediScanPDFReport:
    """PDF report builder class"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
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
    
    def _create_header(self) -> List:
        """Create header section with logo and title"""
        elements = []
        
        # Try to add logo
        logo_path = "static/logo.jpg"
        if os.path.exists(logo_path):
            try:
                logo = Image(logo_path, width=2*inch, height=1*inch)
                elements.append(logo)
                elements.append(Spacer(1, 12))
            except Exception as e:
                print(f"Could not load logo: {e}")
        
        # Main title
        title = Paragraph("🛡️ MediScan AI Chest X-ray Diagnosis Report", self.styles['MediScanHeader'])
        elements.append(title)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_patient_info(self, doctor_name: str, patient_name: str, patient_age: int, 
                           patient_gender: str, urgency_level: str) -> List:
        """Create patient information table"""
        elements = []
        
        # Patient information table
        data = [
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Doctor Name:', doctor_name],
            ['Patient Name:', patient_name],
            ['Patient Age:', f"{patient_age} years"],
            ['Patient Gender:', patient_gender.title()],
            ['Urgency Level:', urgency_level.upper()],
            ['Analysis Method:', 'AI-Powered Deep Learning (DenseNet-121)'],
            ['Conditions Screened:', '14 Thoracic Pathologies']
        ]
        
        # Create table
        table = Table(data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), Color(0.9, 0.9, 0.9)),
            ('TEXTCOLOR', (0, 0), (-1, -1), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [Color(0.95, 0.95, 0.95), Color(1, 1, 1)]),
            ('GRID', (0, 0), (-1, -1), 1, Color(0.8, 0.8, 0.8))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_findings_section(self, findings: List[str]) -> List:
        """Create findings section with categorized results"""
        elements = []
        
        # Section header
        header = Paragraph("Analysis Results", self.styles['MediScanSubHeader'])
        elements.append(header)
        
        if not findings or (len(findings) == 1 and "no major abnormalities" in findings[0].lower()):
            # No abnormalities found
            finding_para = Paragraph(
                "✅ <b>Normal Study:</b> No major abnormalities detected among the 14 thoracic conditions analyzed. "
                "The chest X-ray appears within normal limits for the screened pathologies.",
                self.styles['NormalFinding']
            )
            elements.append(finding_para)
        else:
            # Categorize findings by urgency
            urgent_findings = []
            routine_findings = []
            
            for finding in findings:
                if any(urgent_word in finding.lower() for urgent_word in ['urgent', 'emergency', 'pneumothorax', 'edema']):
                    urgent_findings.append(finding)
                else:
                    routine_findings.append(finding)
            
            # Urgent findings first
            if urgent_findings:
                urgent_header = Paragraph("⚠️ <b>URGENT FINDINGS - Immediate Attention Required:</b>", 
                                        self.styles['UrgentFinding'])
                elements.append(urgent_header)
                
                for finding in urgent_findings:
                    # Clean up finding text
                    clean_finding = finding.replace("•", "").replace("-", "").strip()
                    if clean_finding:
                        finding_para = Paragraph(f"• {clean_finding}", self.styles['UrgentFinding'])
                        elements.append(finding_para)
                
                elements.append(Spacer(1, 10))
            
            # Routine findings
            if routine_findings:
                if urgent_findings:
                    routine_header = Paragraph("📋 <b>Additional Findings:</b>", self.styles['NormalFinding'])
                else:
                    routine_header = Paragraph("🔍 <b>Detected Findings:</b>", self.styles['NormalFinding'])
                
                elements.append(routine_header)
                
                for finding in routine_findings:
                    # Clean up finding text
                    clean_finding = finding.replace("•", "").replace("-", "").strip()
                    if clean_finding and "however, this does not rule out" not in clean_finding.lower():
                        finding_para = Paragraph(f"• {clean_finding}", self.styles['NormalFinding'])
                        elements.append(finding_para)
        
        elements.append(Spacer(1, 15))
        return elements
    
    def _create_image_section(self, heatmap_bytes: bytes) -> List:
        """Create heatmap image section"""
        elements = []
        
        try:
            # Section header
            header = Paragraph("Visual Analysis - Regions of Interest", self.styles['MediScanSubHeader'])
            elements.append(header)
            
            # Process heatmap image
            heatmap_image = PILImage.open(io.BytesIO(heatmap_bytes)).convert("RGB")
            
            # Resize if too large
            max_size = (400, 400)
            heatmap_image.thumbnail(max_size, PILImage.Resampling.LANCZOS)
            
            # Convert to ImageReader
            img_buffer = io.BytesIO()
            heatmap_image.save(img_buffer, format='PNG', quality=95)
            img_buffer.seek(0)
            
            # Add image to document
            image_reader = ImageReader(img_buffer)
            image = Image(image_reader, width=4*inch, height=4*inch)
            elements.append(image)
            
            # Image caption
            caption = Paragraph(
                "<i>Heatmap showing areas of AI model attention. Colored regions indicate areas "
                "where the model detected potential abnormalities. Red/yellow areas show higher "
                "confidence, while blue areas show lower confidence.</i>",
                self.styles['Normal']
            )
            elements.append(caption)
            elements.append(Spacer(1, 15))
            
        except Exception as e:
            print(f"Failed to add heatmap image to PDF: {e}")
            # Add placeholder text
            placeholder = Paragraph(
                "<i>Heatmap visualization could not be generated for this analysis.</i>",
                self.styles['Normal']
            )
            elements.append(placeholder)
            elements.append(Spacer(1, 10))
        
        return elements
    
    def _create_clinical_notes_section(self, clinical_notes: str) -> List:
        """Create clinical notes section"""
        elements = []
        
        header = Paragraph("Clinical Notes", self.styles['MediScanSubHeader'])
        elements.append(header)
        
        # Wrap long text
        wrapped_notes = textwrap.fill(clinical_notes, width=80)
        notes_para = Paragraph(wrapped_notes, self.styles['Normal'])
        elements.append(notes_para)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _create_recommendations_section(self, findings: List[str], urgency_level: str) -> List:
        """Create recommendations section based on findings"""
        elements = []
        
        header = Paragraph("Recommendations", self.styles['MediScanSubHeader'])
        elements.append(header)
        
        # Determine recommendations based on findings and urgency
        recommendations = []
        
        if urgency_level.lower() == 'emergency':
            recommendations.append("🚨 <b>IMMEDIATE ACTION REQUIRED:</b> Seek emergency medical attention immediately.")
        elif urgency_level.lower() == 'urgent':
            recommendations.append("⚠️ <b>URGENT:</b> Schedule immediate consultation with a radiologist or pulmonologist.")
        
        # Check for specific conditions
        has_abnormalities = any(
            "probability" in finding and ":" in finding 
            for finding in findings
        )
        
        if has_abnormalities:
            recommendations.extend([
                "1. <b>Professional Review:</b> Have these AI findings reviewed by a qualified radiologist",
                "2. <b>Clinical Correlation:</b> Consider patient symptoms and clinical history",
                "3. <b>Follow-up Imaging:</b> Additional imaging studies may be recommended",
                "4. <b>Treatment Planning:</b> Develop appropriate treatment plan based on confirmed findings"
            ])
        else:
            recommendations.extend([
                "1. <b>Routine Monitoring:</b> Continue routine chest health monitoring",
                "2. <b>Preventive Care:</b> Maintain regular health check-ups",
                "3. <b>Lifestyle:</b> Follow healthy lifestyle practices for lung health"
            ])
        
        # Add general recommendations
        recommendations.extend([
            "5. <b>Documentation:</b> Keep this report for medical records",
            "6. <b>Follow-up:</b> Schedule appropriate follow-up as recommended by physician"
        ])
        
        for rec in recommendations:
            rec_para = Paragraph(rec, self.styles['Normal'])
            elements.append(rec_para)
            elements.append(Spacer(1, 5))
        
        elements.append(Spacer(1, 15))
        return elements
    
    def _create_signature_section(self) -> List:
        """Create signature section"""
        elements = []
        
        header = Paragraph("Medical Professional Verification", self.styles['MediScanSubHeader'])
        elements.append(header)
        
        # Create signature table
        sig_data = [
            ['Reviewing Physician Signature:', '_' * 40],
            ['Print Name:', '_' * 40],
            ['Date:', '_' * 20],
            ['License Number:', '_' * 25]
        ]
        
        sig_table = Table(sig_data, colWidths=[2*inch, 3*inch])
        sig_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
        ]))
        
        elements.append(sig_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_disclaimer_section(self) -> List:
        """Create medical disclaimer section"""
        elements = []
        
        disclaimer_text = (
            "<b>IMPORTANT MEDICAL DISCLAIMER:</b><br/><br/>"
            "This report contains results from an AI-powered chest X-ray analysis system designed for "
            "screening purposes only. The artificial intelligence model has been trained on chest X-ray "
            "images to detect 14 common thoracic conditions, but it is not a substitute for professional "
            "medical diagnosis.<br/><br/>"
            
            "<b>Key Points:</b><br/>"
            "• AI results require verification by qualified medical professionals<br/>"
            "• This system may produce false positives or false negatives<br/>"
            "• Clinical correlation with patient history and symptoms is essential<br/>"
            "• Additional diagnostic tests may be necessary for definitive diagnosis<br/>"
            "• Emergency cases require immediate medical attention regardless of AI results<br/><br/>"
            
            "<b>Liability:</b> This AI system is a screening tool only. Medical decisions should not be "
            "based solely on these results. Always consult qualified healthcare professionals for "
            "medical advice, diagnosis, and treatment.<br/><br/>"
            
            "<b>Technical Information:</b><br/>"
            "• Model: DenseNet-121 trained on ChestX-ray14 dataset<br/>"
            "• Conditions Screened: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, "
            "Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia<br/>"
            "• Analysis Date: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        disclaimer = Paragraph(disclaimer_text, self.styles['Disclaimer'])
        elements.append(disclaimer)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_footer(self) -> List:
        """Create footer section"""
        elements = []
        
        footer_text = (
            "<b>MediScan AI - Advanced Medical Imaging Analysis</b><br/>"
            "Powered by Deep Learning Technology<br/>"
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
            "For technical support, contact: support@mediscan-ai.com"
        )
        
        footer = Paragraph(footer_text, self.styles['Normal'])
        elements.append(footer)
        
        return elements

# Legacy function for backward compatibility
def generate_pdf_report_legacy(
    doctor_name: str,
    patient_name: str,    
    patient_age: int, 
    patient_gender: str,
    findings: List[str],
    heatmap_bytes: bytes,
    output_path: str = "MediScan_report.pdf",
    signature_line: bool = True
) -> None:
    """Legacy function wrapper"""
    generate_pdf_report(
        doctor_name, patient_name, patient_age, patient_gender,
        findings, heatmap_bytes, output_path, signature_line
    )

if __name__ == "__main__":
    # Test report generation
    test_findings = [
        "Pneumonia: 0.75 probability - Lung infection detected",
        "Mass: 0.60 probability - Abnormal growth in lung tissue"
    ]
    
    try:
        generate_pdf_report(
            "Dr. Smith", "John Doe", 45, "Male",
            test_findings, None, "test_report.pdf"
        )
        print("Test report generated successfully")
    except Exception as e:
        print(f"Test failed: {e}")
