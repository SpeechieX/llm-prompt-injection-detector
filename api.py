from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Prompt Injection Detector")

classifier = pipeline(
  "text-classification",
  model="./model/final"
)

class PromptRequest(BaseModel):
  text: str

class DetectionResult(BaseModel):
  text: str
  is_injection: bool
  confidence: float
  label: str

@app.post("/detect", response_model=DetectionResult)
def detect_injection(request: PromptRequest):
  result = classifier(request.text)[0]
  is_injection = result["label"] == "LABEL_1"
  return DetectionResult(
    text=request.text,
    is_injection=is_injection,
    confidence=round(result["score"], 4),
    label="INJECTION" if is_injection else "BENIGN"
  )

@app.get("/health")
def health():
  return {"status": "ok"}
