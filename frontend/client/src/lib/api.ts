/**
 * VitalWatch API Client
 * Centralized API calls that route through the Express proxy to the FastAPI backend.
 * All paths start with /api/* which the proxy forwards to http://127.0.0.1:8000.
 */

// ─── Types matching backend schemas ─────────────────────────────────────────

export interface Patient {
  id: string;
  name: string;
  age: number;
  condition: string;
  doctor_email: string;
  address?: string;
  created_at?: string;
}

export interface PatientCreate {
  name: string;
  age: number;
  condition: string;
  doctor_email: string;
  address?: string;
}

export interface VitalReading {
  id?: string;
  patient_id: string;
  raw_inputs?: Record<string, number>;
  derived_features?: Record<string, number>;
  risk_score?: number;
  risk_class?: string;
  confidence?: number;
  tier?: string;
  explanation?: string;
  alert_fired?: boolean;
  flags?: string[];
  timestamp?: string;
}

export interface Alert {
  id: string;
  patient_id: string;
  reading_id?: string;
  tier: string;
  status: string;
  fired_at?: string;
  acknowledged: boolean;
}

export interface AmbulanceTicket {
  id: string;
  patient_id: string;
  alert_id?: string;
  status: string; // OPEN | EN_ROUTE | ON_SCENE | RESOLVED
  patient_name?: string;
  patient_address?: string;
  vitals_snapshot?: Record<string, any>;
  risk_score?: number;
  flags?: string[];
  fired_at?: string;
}

export interface TrendPoint {
  timestamp: string;
  risk_score: number;
  heart_rate?: number;
  spo2?: number;
  systolic_bp?: number;
}

export interface PredictResponse {
  risk_score: number;
  risk_class: string;
  confidence: number;
  tier: string;
  flags: string[];
  action: string;
  explanation?: string;
}

export interface FullPredictRequest {
  text: string;
  patient_id: string;
  session_id?: string;
  history?: Record<string, any>[];
}

export interface FullPredictResponse {
  complete: boolean;
  session_id: string;
  missing: string[];
  message: string;
  prediction?: PredictResponse;
  reading_id?: string;
}

export interface HealthStatus {
  status: string;
  models_loaded: boolean;
}

// ─── Base fetch helper ──────────────────────────────────────────────────────

async function apiFetch<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }

  return res.json();
}

// ─── Health ─────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<HealthStatus> {
  return apiFetch<HealthStatus>("/api/health");
}

// ─── Patients ───────────────────────────────────────────────────────────────

export async function getPatients(): Promise<Patient[]> {
  return apiFetch<Patient[]>("/api/patients");
}

export async function getPatient(patientId: string): Promise<Patient> {
  return apiFetch<Patient>(`/api/patients/${patientId}`);
}

export async function createPatient(data: PatientCreate): Promise<{ patient_id: string }> {
  return apiFetch<{ patient_id: string }>("/api/patients", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function getPatientReadings(patientId: string, limit = 30): Promise<VitalReading[]> {
  return apiFetch<VitalReading[]>(`/api/patients/${patientId}/readings?limit=${limit}`);
}

export async function getPatientAlerts(patientId: string): Promise<Alert[]> {
  return apiFetch<Alert[]>(`/api/patients/${patientId}/alerts`);
}

export async function getPatientTrend(patientId: string): Promise<TrendPoint[]> {
  return apiFetch<TrendPoint[]>(`/api/patients/${patientId}/trend`);
}

// ─── Alerts ─────────────────────────────────────────────────────────────────

export async function getActiveAlerts(): Promise<Alert[]> {
  return apiFetch<Alert[]>("/api/alerts/active");
}

export async function acknowledgeAlert(alertId: string): Promise<{ acknowledged: boolean }> {
  return apiFetch<{ acknowledged: boolean }>(`/api/alerts/${alertId}/acknowledge`, {
    method: "PUT",
  });
}

export async function doctorAction(
  alertId: string,
  patientId: string,
  actionType: string,
  note = ""
): Promise<{ action_id: string }> {
  const params = new URLSearchParams({
    action_type: actionType,
    patient_id: patientId,
    note,
  });
  return apiFetch<{ action_id: string }>(`/api/alerts/${alertId}/action?${params}`, {
    method: "POST",
  });
}

// ─── Ambulance Tickets ──────────────────────────────────────────────────────

export async function getOpenTickets(): Promise<AmbulanceTicket[]> {
  return apiFetch<AmbulanceTicket[]>("/api/tickets/open");
}

export async function getAllTickets(): Promise<AmbulanceTicket[]> {
  return apiFetch<AmbulanceTicket[]>("/api/tickets/all");
}

export async function getTicket(ticketId: string): Promise<AmbulanceTicket> {
  return apiFetch<AmbulanceTicket>(`/api/tickets/${ticketId}`);
}

export async function updateTicketStatus(
  ticketId: string,
  status: "EN_ROUTE" | "ON_SCENE" | "RESOLVED"
): Promise<{ ticket_id: string; status: string }> {
  return apiFetch<{ ticket_id: string; status: string }>(`/api/tickets/${ticketId}/status`, {
    method: "PUT",
    body: JSON.stringify({ status }),
  });
}

// ─── Prediction (Chatbot) ───────────────────────────────────────────────────

export async function predictFull(data: FullPredictRequest): Promise<FullPredictResponse> {
  return apiFetch<FullPredictResponse>("/api/predict/full", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

// ─── Input Agent ────────────────────────────────────────────────────────────

export async function parseText(text: string, sessionId?: string) {
  return apiFetch<any>("/api/input/parse-text", {
    method: "POST",
    body: JSON.stringify({ text, session_id: sessionId }),
  });
}
