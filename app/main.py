from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="LLM Data Prep Evaluation API")

app.include_router(router)