from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="Medhā",
    description="AI-powered video-to-interactive learning platform (Phase 8.2: Content Intelligence & Learning Quality Refinement).",
    version="0.8.2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Medhā",
        "phase": "8.2-content-intelligence-quality (includes 8.1)",
        "docs": "/docs",
    }
