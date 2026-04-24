import json
import os
from urllib import error, request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dataclasses import Field

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TriageRequest(BaseModel):
    text: str = Field(min_length=1)

@app.get("/")
def root():
    return {"message": "Backend running"}
# Load trained model and encoders
model = joblib.load("ctas_model.pkl")
le_gender = joblib.load("gender_encoder.pkl")
le_symptoms = joblib.load("symptom_encoder.pkl")
le_disease = joblib.load("disease_encoder.pkl")

# CTAS advice mapping
ctas_recommendation = {
    1: "Go to the Emergency Room immediately!",
    2: "Go to the Emergency Room as soon as possible.",
    3: "Visit a clinic for urgent care.",
    4: "You can go to a clinic or urgent care center.",
    5: "You can manage at home or visit a pharmacy."
}


# Safe transform function (handles unknown values)
def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except:
        return -1
    


@app.post("/api/triage")
def triage(input_data: TriageRequest):
    text = input_data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    webhook_url = os.getenv("N8N_WEBHOOK_URL", "").strip()
    if not webhook_url:
        raise HTTPException(status_code=500, detail="N8N_WEBHOOK_URL is not configured")
    payload = json.dumps({"text": text}).encode("utf-8")
    req = request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=20) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {"reply": ""}
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=502, detail=f"n8n HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Failed calling n8n webhook: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="n8n returned non-JSON response") from exc
    

@app.post("/predict")
def predict(data: dict):

    symptoms = data.get("symptoms", [])
    diseases = data.get("diseases", [])
    location = data.get("location")

    age = data.get("age", 0)
    gender = data.get("gender")
    symptom_count = data.get("symptom_count", len(symptoms))

    # Convert to lowercase for safety
    symptoms = [s.lower() for s in symptoms]

    # ===============================
    # 🚨 CRITICAL EMERGENCY RULES
    # ===============================

    # Heart attack pattern
    if "chest pain" in symptoms and "profuse sweating" in symptoms:
        return {
            "predicted_ctas": 1,
            "advice": "Possible heart attack. Go to the Emergency Room immediately!",
            "location": location
        }

    if "chest pain" in symptoms and "vomiting" in symptoms:
        return {
            "predicted_ctas": 1,
            "advice": "Possible cardiac emergency. Please go to the Emergency Room immediately.",
            "location": location
        }

    # Possible concussion
    if "head injury" in symptoms and "dizziness" in symptoms:
        return {
            "predicted_ctas": 2,
            "advice": "Possible concussion. Go to the Emergency Room as soon as possible.",
            "location": location
        }

    # Severe breathing issue
    if "shortness of breath" in symptoms and "chest pain" in symptoms:
        return {
            "predicted_ctas": 1,
            "advice": "Severe breathing difficulty detected. Go to the Emergency Room immediately.",
            "location": location
        }

    # ===============================
    # 🤖 MACHINE LEARNING PREDICTION
    # ===============================

    symptom_string = ",".join(symptoms)
    disease_string = ",".join(diseases)

    gender_encoded = safe_transform(le_gender, gender)
    symptom_encoded = safe_transform(le_symptoms, symptom_string)
    disease_encoded = safe_transform(le_disease, disease_string)

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender_encoded,
        "Symptoms": symptom_encoded,
        "Symptom_Count": symptom_count,
        "Disease": disease_encoded
    }])

    prediction = model.predict(input_data)
    ctas_level = int(prediction[0])

    return {
        "predicted_ctas": ctas_level,
        "advice": ctas_recommendation.get(ctas_level),
        "location": location
    }
