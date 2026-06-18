# VitalWatch: Post-Discharge Vital Monitoring and AI-Powered Alert System

VitalWatch is an AI-powered, clinician-first clinical telemetry and patient monitoring platform. It is designed to track post-discharge patients, run clinical rule engines and machine learning classifiers to assess patient risk tiers, generate clinical explanations, and automate urgent medical notifications via Email, Push, and Emergency Medical Services (EMS) Ambulance Tickets.

## Key Features

* Multi-Agent AI Pipeline
  * Input Agent: Extracts vital signs from patient-reported free-form text or logs using conversational state tracking persisted in Redis. Supports multi-turn patient inputs if some parameters are missing.
  * Explanation Agent: Generates clinician-friendly, structured explanations for calculated risk scores and abnormal readings.
* Machine Learning Inference
  * Continuous Risk Regression: Uses an XGBoost Regressor to calculate the patient risk index.
  * Confidence Classification: Uses a LightGBM Classifier to categorize patients into Good, Ambiguous, and Bad risk classes.
* Dynamic Feature Engineering
  * Computes derived clinical metrics such as Pulse Pressure, Shock Index, Oxygen Deficit, Heart Rate/Temperature Ratio, and rolling averages/historical trend metrics.
* Automated Routing and Alerts
  * NORMAL (Risk < 0.06): Safe state, silently logged.
  * WARNING (0.06 <= Risk < 0.30): High-priority email alert sent to the doctor via Resend.
  * CRITICAL (Risk >= 0.30): Doctor email alert via Resend, push notification to family via ntfy.sh, and automatic dispatch ticket created for emergency ambulance services in Supabase.
* Clinician Dashboard Integration
  * Restful endpoints for patients database, alerts history, ambulance dispatch state, and doctor actions logs.

## Tech Stack

* Backend: FastAPI (Python 3.10+)
* LLM Engine: Groq SDK (llama-3.3-70b-versatile)
* State Store: Upstash Redis (Serverless Redis)
* Database: Supabase (PostgreSQL client)
* ML Models: XGBoost, LightGBM, Joblib, Scikit-Learn
* Notification Services: Resend (Emails), ntfy.sh (Push Notifications)
* Frontend: React, Vite, TailwindCSS, Express Server

## Installation and Setup

### 1. Prerequisites
* Python 3.10 or 3.11
* Node.js (v18 or higher)
* Git

### 2. Clone the Repository
```bash
git clone https://github.com/aryan9-6-5/VitalWatch
cd VitalWatch
```

---

## Backend Configuration and Run

### 1. Initialize Virtual Environment
Navigate to the root directory and create a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the `backend/` directory and configure the following parameters:
```env
# Groq API Configuration
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=llama-3.3-70b-versatile

# Supabase Database Connection
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-key

# Upstash Redis Configuration (Chatbot Session State)
UPSTASH_REDIS_REST_URL=your-upstash-redis-rest-url
UPSTASH_REDIS_REST_TOKEN=your-upstash-redis-rest-token

# Alert Gateways
RESEND_API_KEY=your-resend-api-key
ALERT_EMAIL_FROM=alerts@resend.dev
NTFY_TOPIC=vitalwatch-alerts

# Risk Settings
RISK_CRITICAL_THRESHOLD=0.30
RISK_WARNING_THRESHOLD=0.06
```

### 3. Running the Backend Server
From the `backend/` directory, launch the Uvicorn server:
```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Once started, the interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## Frontend Configuration and Run

The frontend is a React application served via a custom Express server that handles reverse-proxying API queries to the backend.

### 1. Install Dependencies
Navigate to the `frontend/` directory and install the packages:
```bash
cd frontend
npm install
```

### 2. Running the Frontend Server
Run the development server. Make sure the environment variable `NODE_ENV` is set to `development`:

On Windows (PowerShell):
```powershell
$env:NODE_ENV="development"; npx tsx server/index.ts
```

On Linux/macOS:
```bash
NODE_ENV=development npx tsx server/index.ts
```

The application will be served locally at `http://localhost:5000`.

---

## API Endpoint Routing

All frontend API calls are routed through the proxy `/api/*` prefix:

* ML Inference Pipeline:
  * POST `/api/predict`: Direct prediction interface. Takes 10 raw parameters plus history, returns risk assessment immediately.
  * POST `/api/predict/full`: Chatbot pipeline. Processes raw user message, parses vitals (with Redis session support), calculates risk, triggers notification alerts, writes to database, and responds.
* Patients and Trends:
  * POST `/api/patients`: Create a patient registry.
  * GET `/api/patients`: Retrieve list of all registered patients.
  * GET `/api/patients/{patient_id}`: Get specific patient details.
  * GET `/api/patients/{patient_id}/readings`: Get patient's historical vitals.
  * GET `/api/patients/{patient_id}/trend`: Get 7-day risk, Heart Rate, and SpO2 trend data (formatted for frontend charts).
* Alerting and Emergency:
  * GET `/api/alerts/active`: Retrieve unacknowledged medical alerts.
  * PUT `/api/alerts/{alert_id}/acknowledge`: Clinicians acknowledge alerts.
  * POST `/api/alerts/{alert_id}/action`: Record clinical action taken.
  * GET `/api/tickets/open`: Retrieve open ambulance dispatch tickets.
  * PUT `/api/tickets/{ticket_id}/status`: Update paramedic status (EN_ROUTE, ON_SCENE, RESOLVED).
