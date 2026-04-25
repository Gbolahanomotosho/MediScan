# 🩺 MediScan – AI-Powered Medical Image Analysis

**MediScan** is an advanced AI web application that analyzes medical images (X-rays, CT scans, MRIs) to detect abnormalities and assist radiologists with preliminary diagnoses.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-lightgrey)
![Render](https://img.shields.io/badge/Render-Deployment-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Multi-Disease Detection** | Analyzes images for pneumonia, fractures, tumors, and more |
| 📄 **Automated PDF Report** | Generates comprehensive diagnostic reports |
| 📧 **Email Results** | Send reports to physicians or patients |
| 🗄️ **Case Management** | SQLite database stores all analyses for audit trails |
| ☁️ **Cloud Ready** | Configured for deployment on Render (see `render.yaml` & `Procfile`) |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Deep Learning** | PyTorch, custom trained model (`model.pth.tar`) |
| **Backend** | Python, Flask, Gunicorn |
| **Database** | SQLite |
| **Frontend** | HTML, CSS (in `templates/` & `static/`) |
| **Utilities** | ReportLab (PDF), `smtplib` (email) |
| **Deployment** | Render (configuration: `render.yaml`, `Procfile`) |

---

## 📂 Project Structure

```

MediScan/
├── app.py                # Main Flask application
├── analyze.py            # PyTorch model inference logic
├── database.py           # SQLite operations
├── pdf_report.py         # PDF report generator
├── email_sender.py       # Email automation
├── requirements.txt      # Python dependencies
├── Procfile              # Gunicorn entry for Render
├── render.yaml           # Render deployment config
├── model.pth.tar         # Trained PyTorch model weights
├── reports/              # Generated PDF reports
├── static/               # CSS, JavaScript, images
└── templates/            # HTML templates

```

---

## 🔧 Installation & Local Testing

```bash
# Clone the repository
git clone https://github.com/Gbolahanomotosho/MediScan.git
cd MediScan

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py

# Open your browser to http://127.0.0.1:5000
```

---

🧠 What I Built (My Contribution)

· Custom PyTorch model – Trained on medical imaging dataset for multi-class classification
· Complete Flask web application – Upload, predict, display confidence scores
· Automated PDF report generation – Professional medical reports with findings
· Email integration – Send reports with secure attachments
· SQLite database – Track and audit all predictions
· Production deployment – Configured for Render cloud platform
· Responsive UI – Clean interface for medical professionals

---

📈 Example Workflow

1. User uploads medical image (X-ray, CT, MRI)
2. MediScan runs PyTorch inference using model.pth.tar
3. Results displayed – predicted condition + confidence percentage
4. PDF report generated – includes patient details, findings, and images
5. Optionally emailed to specified doctor/patient
6. Case saved to SQLite database for future reference

---

🚧 Current Status & Planned Improvements

Component Status
Multi-disease classification ✅ Complete
Web interface ✅ Complete
PDF report generation ✅ Complete
Email functionality ✅ Complete
Database storage ✅ Complete
Render deployment ✅ Complete
Model accuracy improvement 🔄 Planned (needs more training data)
Heatmap visualization (Grad-CAM) 🔄 Planned
DICOM format support 🔄 Planned
German language reports 🔄 Planned
HIPAA/GDPR compliance 🔄 Planned

---

🏥 Why This Matters for German Employers

This project demonstrates:

· ✅ Deep Learning (PyTorch) – Custom model training for medical domain
· ✅ Full-stack deployment – Flask, database, cloud ready
· ✅ Real-world healthcare utility – Assists radiologists with preliminary reads
· ✅ Integration skills – Email, PDF, database, web interface
· ✅ Production mindset – Deployed on Render with proper configuration files

---

📫 Contact & Visa Status

Omotosho Gbolahan Hammed

· GitHub: Gbolahanomotosho
· Email: hammedg621@gmail.com
· 🛂 German IT Specialist Visa Eligible – 7+ years IT experience. No degree recognition required.

---

📜 License

MIT License – free for academic and commercial use with attribution.
