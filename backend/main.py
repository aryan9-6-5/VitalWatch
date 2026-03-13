from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings
from services.inference import load_models, models_loaded
from routers import predict, input, alerts, patients, tickets


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    print("🚀 VitalWatch backend starting...")
    success = load_models()
    if success:
        print("✅ All ML models loaded and ready")
    else:
        print("⚠️  Model loading failed — /api/predict will return 503")
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    print("VitalWatch backend shutting down")


app = FastAPI(
    title       = "VitalWatch API",
    description = "Post-discharge vital monitoring — AI-powered alert system",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = [settings.FRONTEND_URL, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(input.router)
app.include_router(alerts.router)
app.include_router(patients.router)
app.include_router(tickets.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    loaded = models_loaded()
    return {
        "status":        "ok" if loaded else "degraded",
        "models_loaded": loaded,
    }


@app.get("/")
async def root():
    return {"message": "VitalWatch API v1.0 — visit /docs for API reference"}