from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # Upstash Redis
    UPSTASH_REDIS_REST_URL: str = ""
    UPSTASH_REDIS_REST_TOKEN: str = ""

    # Alerts
    RESEND_API_KEY: str = ""
    NTFY_TOPIC: str = "vitalwatch-alerts"
    ALERT_EMAIL_FROM: str = "alerts@resend.dev"

    # Risk thresholds
    RISK_CRITICAL_THRESHOLD: float = 0.70
    RISK_WARNING_THRESHOLD: float = 0.40

    # App
    SECRET_KEY: str = "change-me-in-production-min-32-chars"
    FRONTEND_URL: str = "http://localhost:5173"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()