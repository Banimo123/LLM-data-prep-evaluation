from fastapi import FastAPI

app = FastAPI(title="LLM Data Prep Evaluation API")

@app.get("/health")
def health_check():
    return {"status": "ok"}