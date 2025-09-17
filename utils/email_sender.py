# email_sender.py - Enhanced Email System for MediScan AI
import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email import encoders
from dotenv import load_dotenv
from typing import Optional, List
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Email configuration
EMAIL_CONFIG = {
    'address': os.getenv("EMAIL_ADDRESS"),
    'password': os.getenv("EMAIL_PASSWORD"), 
    'smtp_server': os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    'smtp_port': int(os.getenv("SMTP_PORT", "587")),
    'use_tls': os.getenv("USE_TLS", "true").lower() == "true"
}

class MediScanEmailSender:
    """Advanced email sender for MediScan AI reports"""
    
    def __init__(self):
        self.config = EMAIL_CONFIG
        self.validate_config()
    
    def validate_config(self):
        """Validate email configuration"""
        required_fields = ['address', 'password', 'smtp_server']
        missing_fields = [field for field in required_fields if not self.config.get(field)]
        
        if missing_fields:
            logger.warning(f"Missing email configuration: {missing_fields}")
            raise ValueError(f"Missing email configuration: {missing_fields}")
    
    def create_professional_email(
        self,
        recipient_email: str,
        patient_name: str,
        doctor_name: str,
        report_id: str,
        findings_summary: str,
        urgency_level: str = "routine"
    ) -> MIMEMultipart:
        """Create a professional medical report email"""
        
        msg = MIMEMultipart('alternative')
        
        # Email headers
        msg['Subject'] = f"MediScan AI - Chest X-ray Analysis Report (ID: {report_id})"
        msg['From'] = self.config['address']
        msg['To'] = recipient_email
        msg['Reply-To'] = self.config['address']
        
        # Add priority based on urgency
        if urgency_level.lower() == 'emergency':
            msg['X-Priority'] = '1'
            msg['X-MSMail-Priority'] = 'High'
            msg['Importance'] = 'High'
        elif urgency_level.lower() == 'urgent':
            msg['X-Priority'] = '2'
            msg['X-MSMail-Priority'] = 'High'
        
        # Create HTML email content
        html_content = self._create_html_email(
            patient_name, doctor_name, report_id, findings_summary, urgency_level
        )
        
        # Create plain text version
        text_content = self._create_text_email(
            patient_name, doctor_name, report_id, findings_summary, urgency_level
        )
        
        # Attach both versions
        msg.attach(MIMEText(text_content, 'plain', 'utf-8'))
        msg.attach(MIMEText(html_content, 'html', 'utf-8'))
        
        return msg
    
    def _create_html_email(
        self, patient_name: str, doctor_name: str, report_id: str, 
        findings_summary: str, urgency_level: str
    ) -> str:
        """Create HTML email content"""
        
        urgency_color = {
            'emergency': '#ef4444',
            'urgent': '#f59e0b', 
            'routine': '#10b981'
        }.get(urgency_level.lower(), '#10b981')
        
        urgency_icon = {
            'emergency': '🚨',
            'urgent': '⚠️',
            'routine': '📋'
        }.get(urgency_level.lower(), '📋')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MediScan AI Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f8fafc; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                .header h1 {{ margin: 0; font-size: 24px; font-weight: 600; }}
                .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
                .content {{ padding: 30px; }}
                .urgency-badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-size: 12px; font-weight: 600; text-transform: uppercase; color: white; background-color: {urgency_color}; margin-bottom: 20px; }}
                .info-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .info-table td {{ padding: 12px; border-bottom: 1px solid #e5e7eb; }}
                .info-table td:first-child {{ font-weight: 600; color: #374151; background-color: #f9fafb; width: 30%; }}
                .findings {{ background-color: #f0f9ff; border-left: 4px solid #3b82f6; padding: 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
                .footer {{ background-color: #f8fafc; padding: 20px; text-align: center; border-top: 1px solid #e5e7eb; }}
                .footer p {{ margin: 5px 0; font-size: 14px; color: #6b7280; }}
                .disclaimer {{ background-color: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 16px; margin: 20px 0; }}
                .disclaimer h4 {{ margin: 0 0 8px 0; color: #92400e; }}
                .disclaimer p {{ margin: 0; font-size: 13px; color: #451a03; }}
                .button {{ display: inline-block; background-color: #10b981; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 600; margin: 10px 0; }}
                .button:hover {{ background-color: #059669; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🛡️ MediScan AI</h1>
                    <p>Chest X-ray Analysis Report</p>
                </div>
                
                <div class="content">
                    <div class="urgency-badge">
                        {urgency_icon} {urgency_level.upper()} PRIORITY
                    </div>
                    
                    <p>Dear {patient_name},</p>
                    
                    <p>Your chest X-ray analysis has been completed using our advanced AI diagnostic system. Please find your comprehensive report attached to this email.</p>
                    
                    <table class="info-table">
                        <tr>
                            <td>Patient Name:</td>
                            <td>{patient_name}</td>
                        </tr>
                        <tr>
                            <td>Attending Doctor:</td>
                            <td>{doctor_name}</td>
                        </tr>
                        <tr>
                            <td>Report ID:</td>
                            <td>{report_id}</td>
                        </tr>
                        <tr>
                            <td>Analysis Date:</td>
                            <td>{time.strftime('%Y-%m-%d %H:%M:%S')}</td>
                        </tr>
                        <tr>
                            <td>Priority Level:</td>
                            <td><span style="color: {urgency_color}; font-weight: 600;">{urgency_level.upper()}</span></td>
                        </tr>
                    </table>
                    
                    <div class="findings">
                        <h3 style="margin-top: 0; color: #1e40af;">Analysis Summary</h3>
                        <p>{findings_summary}</p>
                    </div>
                    
                    {"<div style='background-color: #fee2e2; border: 1px solid #f87171; border-radius: 8px; padding: 16px; margin: 20px 0;'><h4 style='margin: 0 0 8px 0; color: #dc2626;'>⚠️ URGENT ATTENTION REQUIRED</h4><p style='margin: 0; color: #7f1d1d;'>This report indicates findings that may require immediate medical attention. Please contact your healthcare provider immediately.</p></div>" if urgency_level.lower() in ['emergency', 'urgent'] else ""}
                    
                    <div class="disclaimer">
                        <h4>⚠️ Important Medical Disclaimer</h4>
                        <p>This AI analysis is for screening purposes only and should not replace professional medical diagnosis. All findings must be verified by qualified medical professionals. Always consult your healthcare provider for medical advice.</p>
                    </div>
                    
                    <p><strong>Next Steps:</strong></p>
                    <ul>
                        <li>Review the attached detailed PDF report</li>
                        <li>Share these results with your healthcare provider</li>
                        <li>{"Seek immediate medical attention" if urgency_level.lower() == 'emergency' else "Schedule a follow-up appointment as recommended"}</li>
                        <li>Keep this report for your medical records</li>
                    </ul>
                    
                    <p>If you have any questions about this report, please contact your healthcare provider immediately.</p>
                    
                    <p>Best regards,<br>
                    <strong>MediScan AI Team</strong></p>
                </div>
                
                <div class="footer">
                    <p><strong>MediScan AI - Advanced Medical Imaging Analysis</strong></p>
                    <p>Powered by Deep Learning Technology</p>
                    <p>This is an automated message. Please do not reply to this email.</p>
                    <p>For technical support: support@mediscan-ai.com</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_text_email(
        self, patient_name: str, doctor_name: str, report_id: str,
        findings_summary: str, urgency_level: str
    ) -> str:
        """Create plain text email content"""
        
        urgency_icon = {
            'emergency': '[EMERGENCY]',
            'urgent': '[URGENT]', 
            'routine': '[ROUTINE]'
        }.get(urgency_level.lower(), '[ROUTINE]')
        
        text = f"""
MediScan AI - Chest X-ray Analysis Report {urgency_icon}

Dear {patient_name},

Your chest X-ray analysis has been completed using our advanced AI diagnostic system. Please find your comprehensive report attached to this email.

REPORT DETAILS:
- Patient Name: {patient_name}
- Attending Doctor: {doctor_name}
- Report ID: {report_id}
- Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Priority Level: {urgency_level.upper()}

ANALYSIS SUMMARY:
{findings_summary}

{"="*50}
{"⚠️  URGENT ATTENTION REQUIRED" if urgency_level.lower() in ['emergency', 'urgent'] else ""}
{"This report indicates findings that may require immediate medical attention." if urgency_level.lower() in ['emergency', 'urgent'] else ""}
{"Please contact your healthcare provider immediately." if urgency_level.lower() in ['emergency', 'urgent'] else ""}
{"="*50 if urgency_level.lower() in ['emergency', 'urgent'] else ""}

IMPORTANT MEDICAL DISCLAIMER:
This AI analysis is for screening purposes only and should not replace 
professional medical diagnosis. All findings must be verified by qualified 
medical professionals. Always consult your healthcare provider for medical advice.

NEXT STEPS:
1. Review the attached detailed PDF report
2. Share these results with your healthcare provider
3. {"Seek immediate medical attention" if urgency_level.lower() == 'emergency' else "Schedule a follow-up appointment as recommended"}
4. Keep this report for your medical records

If you have any questions about this report, please contact your healthcare 
provider immediately.

Best regards,
MediScan AI Team

---
MediScan AI - Advanced Medical Imaging Analysis
Powered by Deep Learning Technology
This is an automated message. Please do not reply to this email.
For technical support: support@mediscan-ai.com
        """
        
        return text.strip()
    
    def send_report_email(
        self,
        recipient_email: str,
        pdf_path: str,
        patient_name: str = "Patient",
        doctor_name: str = "Doctor",
        report_id: str = "N/A",
        findings_summary: str = "Analysis completed",
        urgency_level: str = "routine",
        max_retries: int = 3
    ) -> bool:
        """
        Send medical report email with PDF attachment
        
        Args:
            recipient_email: Email address to send to
            pdf_path: Path to PDF report file
            patient_name: Patient's name
            doctor_name: Doctor's name
            report_id: Unique report identifier
            findings_summary: Summary of findings
            urgency_level: Priority level (routine, urgent, emergency)
            max_retries: Maximum number of send attempts
            
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        
        if not self.config['address'] or not self.config['password']:
            logger.error("Email credentials not configured")
            return False
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending report email to {recipient_email} (attempt {attempt + 1})")
                
                # Create email message
                msg = self.create_professional_email(
                    recipient_email, patient_name, doctor_name, 
                    report_id, findings_summary, urgency_level
                )
                
                # Attach PDF report
                with open(pdf_path, "rb") as attachment:
                    pdf_part = MIMEApplication(attachment.read(), _subtype="pdf")
                    pdf_part.add_header(
                        "Content-Disposition",
                        f"attachment; filename=MediScan_Report_{report_id}.pdf"
                    )
                    msg.attach(pdf_part)
                
                # Send email
                with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                    if self.config['use_tls']:
                        server.starttls()
                    
                    server.login(self.config['address'], self.config['password'])
                    server.send_message(msg)
                
                logger.info(f"Email sent successfully to {recipient_email}")
                return True
                
            except smtplib.SMTPAuthenticationError:
                logger.error("SMTP authentication failed - check credentials")
                return False
                
            except smtplib.SMTPRecipientsRefused:
                logger.error(f"Recipient email refused: {recipient_email}")
                return False
                
            except Exception as e:
                logger.error(f"Email sending attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("All email sending attempts failed")
                    return False
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def test_connection(self) -> bool:
        """Test email server connection"""
        try:
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                if self.config['use_tls']:
                    server.starttls()
                server.login(self.config['address'], self.config['password'])
                logger.info("Email connection test successful")
                return True
        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False

# Global email sender instance
_email_sender = None

def get_email_sender() -> MediScanEmailSender:
    """Get or create global email sender instance"""
    global _email_sender
    if _email_sender is None:
        try:
            _email_sender = MediScanEmailSender()
        except Exception as e:
            logger.error(f"Failed to initialize email sender: {e}")
            raise
    return _email_sender

def send_email(
    recipient_email: str, 
    pdf_path: str,
    patient_name: str = "Patient",
    doctor_name: str = "Doctor", 
    report_id: str = "N/A",
    findings_summary: str = "Analysis completed",
    urgency_level: str = "routine"
) -> bool:
    """
    Main email sending function for backward compatibility
    
    Args:
        recipient_email: Email address to send to
        pdf_path: Path to PDF report
        patient_name: Patient's name
        doctor_name: Doctor's name
        report_id: Report identifier
        findings_summary: Summary of findings
        urgency_level: Priority level
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        sender = get_email_sender()
        return sender.send_report_email(
            recipient_email=recipient_email,
            pdf_path=pdf_path,
            patient_name=patient_name,
            doctor_name=doctor_name,
            report_id=report_id,
            findings_summary=findings_summary,
            urgency_level=urgency_level
        )
    except Exception as e:
        logger.error(f"Email sending failed: {e}")
        return False

def send_notification_email(
    recipient_email: str,
    subject: str,
    message: str,
    urgency_level: str = "routine"
) -> bool:
    """Send a simple notification email"""
    try:
        sender = get_email_sender()
        
        msg = MIMEMultipart()
        msg['Subject'] = f"MediScan AI - {subject}"
        msg['From'] = sender.config['address']
        msg['To'] = recipient_email
        
        # Add priority headers
        if urgency_level.lower() == 'emergency':
            msg['X-Priority'] = '1'
            msg['X-MSMail-Priority'] = 'High'
        
        msg.attach(MIMEText(message, 'plain', 'utf-8'))
        
        with smtplib.SMTP(sender.config['smtp_server'], sender.config['smtp_port']) as server:
            if sender.config['use_tls']:
                server.starttls()
            server.login(sender.config['address'], sender.config['password'])
            server.send_message(msg)
        
        return True
        
    except Exception as e:
        logger.error(f"Notification email failed: {e}")
        return False

def check_email_configuration() -> dict:
    """Check email configuration status"""
    try:
        sender = get_email_sender()
        connection_ok = sender.test_connection()
        
        return {
            "configured": True,
            "connection_test": connection_ok,
            "smtp_server": sender.config['smtp_server'],
            "smtp_port": sender.config['smtp_port'],
            "email_address": sender.config['address']
        }
    except Exception as e:
        return {
            "configured": False,
            "error": str(e),
            "connection_test": False
        }

if __name__ == "__main__":
    # Test email configuration
    config_status = check_email_configuration()
    print("Email Configuration Status:")
    print(f"Configured: {config_status['configured']}")
    
    if config_status['configured']:
        print(f"SMTP Server: {config_status['smtp_server']}")
        print(f"SMTP Port: {config_status['smtp_port']}")
        print(f"Email Address: {config_status['email_address']}")
        print(f"Connection Test: {'✅ Passed' if config_status['connection_test'] else '❌ Failed'}")
        
        if config_status['connection_test']:
            print("\n✅ Email system ready for sending reports")
        else:
            print("\n❌ Email system not ready - check credentials")
    else:
        print(f"❌ Configuration Error: {config_status.get('error', 'Unknown error')}")
        print("\nPlease ensure the following environment variables are set:")
        print("- EMAIL_ADDRESS")
        print("- EMAIL_PASSWORD") 
        print("- SMTP_SERVER (optional, defaults to smtp.gmail.com)")
        print("- SMTP_PORT (optional, defaults to 587)")
