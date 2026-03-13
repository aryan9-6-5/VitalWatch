from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


# ─── Raw vital inputs (10 params) ───────────────────────────────────────────

class RawVitals(BaseModel):
    systolic_bp: float = Field(..., ge=60, le=250, description="Systolic BP in mmHg")
    diastolic_bp: float = Field(..., ge=40, le=150, description="Diastolic BP in mmHg")
    heart_rate: float = Field(..., ge=30, le=220, description="Heart rate in bpm")
    spo2: float = Field(..., ge=70, le=100, description="SpO2 in %")
    temperature: float = Field(..., ge=34.0, le=42.0, description="Body temp in °C")
    respiratory_rate: float = Field(..., ge=5, le=60, description="Breaths per min")
    ecg: float = Field(..., ge=0.3, le=2.0, description="ECG amplitude in mV")
    cardiac_output: float = Field(..., ge=1.0, le=15.0, description="Cardiac output L/min")
    steps: float = Field(default=0, ge=0, description="Daily steps (optional)")
    calories: float = Field(default=0, ge=0, description="Daily calories (optional)")


# ─── Prediction ─────────────────────────────────────────────────────────────

class PredictRequest(RawVitals):
    patient_id: str
    history: Optional[List[dict]] = []


class PredictResponse(BaseModel):
    risk_score: float
    risk_class: str          # Good / Ambiguous / Bad
    confidence: float
    tier: str                # NORMAL / WARNING / CRITICAL
    flags: List[str]
    action: str
    explanation: Optional[str] = None


# ─── Input agent ─────────────────────────────────────────────────────────────

class ParseTextRequest(BaseModel):
    text: str
    session_id: Optional[str] = None


class ParsedParams(BaseModel):
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    temperature: Optional[float] = None
    respiratory_rate: Optional[float] = None
    ecg: Optional[float] = None
    cardiac_output: Optional[float] = None
    steps: Optional[float] = None
    calories: Optional[float] = None


class ParseTextResponse(BaseModel):
    session_id: str
    extracted: ParsedParams
    missing: List[str]
    complete: bool
    message: str             # what to show in the chatbot


# ─── Full one-shot endpoint ──────────────────────────────────────────────────

class FullPredictRequest(BaseModel):
    text: str                # raw input text from chatbot
    patient_id: str
    session_id: Optional[str] = None
    history: Optional[List[dict]] = []


class FullPredictResponse(BaseModel):
    complete: bool
    session_id: str
    missing: List[str]
    message: str             # chatbot reply
    prediction: Optional[PredictResponse] = None
    reading_id: Optional[str] = None


# ─── Patients ────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    name: str
    age: int
    condition: str
    doctor_email: str
    address: Optional[str] = ""


class PatientResponse(BaseModel):
    id: str
    name: str
    age: int
    condition: str
    doctor_email: str
    address: Optional[str] = ""
    created_at: Optional[datetime] = None


# ─── Alerts ──────────────────────────────────────────────────────────────────

class AlertResponse(BaseModel):
    id: str
    patient_id: str
    reading_id: Optional[str] = None
    tier: str
    status: str
    fired_at: Optional[datetime] = None
    acknowledged: bool = False


# ─── Ambulance tickets ───────────────────────────────────────────────────────

class TicketResponse(BaseModel):
    id: str
    patient_id: str
    alert_id: Optional[str] = None
    status: str              # OPEN / EN_ROUTE / ON_SCENE / RESOLVED
    patient_name: Optional[str] = None
    patient_address: Optional[str] = None
    vitals_snapshot: Optional[dict] = None
    risk_score: Optional[float] = None
    flags: Optional[List[str]] = []
    fired_at: Optional[datetime] = None


class TicketStatusUpdate(BaseModel):
    status: str              # EN_ROUTE / ON_SCENE / RESOLVED


# ─── Doctor actions ──────────────────────────────────────────────────────────

class DoctorAction(BaseModel):
    alert_id: str
    action_type: str         # test / visit / increase_monitoring
    note: Optional[str] = ""


# ─── Health check ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    regressor: bool
    classifier: bool
    scaler: bool