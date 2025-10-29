from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from twilio.rest import Client
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="CareBloom - AI-Powered Pregnancy Risk Assessment API")

# ===========================
# CORS Configuration
# ===========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://carebloom-gamma.vercel.app/","http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# ===========================
# Load Models and Scalers
# ===========================
try:
    model_a = joblib.load("model_a.joblib")
    scaler_a = joblib.load("scaler_model_a.joblib")
    print("✅ Model A and Scaler A loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load Model A or Scaler A: {e}")

# Try to load Model B (optional for disease detection)
try:
    model_b = joblib.load("model_b.joblib")
    scaler_b = joblib.load("scaler_model_b.joblib")
    print("✅ Model B and Scaler B loaded successfully")
    MODEL_B_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Warning: Model B not found. Disease detection will be disabled. {e}")
    model_b = None
    scaler_b = None
    MODEL_B_AVAILABLE = False

# ===========================
# Twilio Config
# ===========================
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_NUMBER = os.getenv("TO_NUMBER", "whatsapp:+923000976116")

if not TWILIO_SID or not TWILIO_AUTH_TOKEN:
    print("⚠️ Warning: Twilio credentials not found in environment variables")
    print("WhatsApp alerts will be disabled")
    client = None
else:
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# ===========================
# Groq Setup
# ===========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# ===========================
# Input Schema
# ===========================
class InputData(BaseModel):
    Age: float
    SystolicBP: float
    DiastolicBP: float
    BS: float
    BodyTemp: float
    HeartRate: float
    PulsePressure: float
    gravida: float = 0
    parity: float = 0
    gestational_age_weeks: float = 0
    Age_yrs: float = 0
    BMI: float = 0
    diabetes: int = 0
    hypertension: int = 0
    HB: float = 0
    fetal_weight: float = 0
    Protien_Uria: int = 0
    amniotic_fluid_levels: float = 0


class AlertData(BaseModel):
    Risk_Level: str
    phone_number: str = None  # Optional phone number


# ===========================
# WhatsApp Alert Function
# ===========================
def send_whatsapp_alert(risk_level: str, phone_number: str = None):
    if not client:
        print("⚠️ Twilio client not configured. Alert not sent.")
        return None
    
    # Use provided phone number or fall back to default
    target_number = f"whatsapp:{phone_number}" if phone_number else TO_NUMBER
    print(f"📱 Sending WhatsApp alert for {risk_level} to {target_number}")
        
    if risk_level == "high risk":
        body = "🚨 *High Risk Alert!* Please contact your doctor immediately for a detailed checkup."
    elif risk_level == "mid risk":
        body = "⚠️ *Warning:* Your pregnancy shows moderate risk. Take care and schedule a medical check soon."
    else:
        body = "✅ *Safe:* Your pregnancy risk is low. Keep following a healthy lifestyle!"

    try:
        message = client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            body=body,
            to=target_number
        )
        print(f"✅ WhatsApp message sent successfully! SID: {message.sid}")
        return message.sid
    except Exception as e:
        print(f"❌ Failed to send WhatsApp message: {e}")
        return None


# ===========================
# AI Advice Generator
# ===========================
def generate_advice(risk_level: str, disease_status: str):
    prompt = f"""
You are a professional pregnancy health assistant.
Based on these conditions:
- Pregnancy Risk Level: {risk_level}
- Disease Status: {disease_status}

Give a short, clear, and empathetic medical advice (2–4 sentences)
for the patient, including recommendations or precautions.
Avoid technical words. Example tone:
'You are doing great! Keep up healthy habits and attend regular checkups.'
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=120
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Failed to generate advice: {e}"


# ===========================
# Health Check Endpoint
# ===========================
@app.get("/")
def health_check():
    return {
        "status": "✅ CareBloom API is running!",
        "endpoints": {
            "predict": "/predict/ (POST)",
            "send_alert": "/send_alert/ (POST)",
            "docs": "/docs (GET)",
            "health": "/ (GET)"
        },
        "models_loaded": {
            "model_a": "✅ Loaded",
            "model_b": "✅ Loaded" if MODEL_B_AVAILABLE else "⚠️ Not available (disease detection disabled)"
        },
        "twilio_configured": "✅ Configured" if client else "⚠️ Not configured"
    }


# ===========================
# Prediction Endpoint
# ===========================
@app.post("/predict/")
def predict(data: InputData):
    try:
        body_temp_fahrenheit = (data.BodyTemp * 9 / 5) + 32

        x_a = pd.DataFrame(
            [[data.Age, data.SystolicBP, data.DiastolicBP, data.BS,
              body_temp_fahrenheit, data.HeartRate, data.PulsePressure]],
            columns=["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "PulsePressure"]
        )

        x_a_scaled = scaler_a.transform(x_a)
        risk_pred = int(model_a.predict(x_a_scaled)[0])
        risk_map = {1: "low risk", 2: "mid risk", 0: "high risk"}
        risk_label = risk_map.get(risk_pred, "unknown")

        result = {"Risk_Level": risk_label}

        # Only run Model B for high risk cases
        if risk_label == "high risk":
            if MODEL_B_AVAILABLE:
                feature_map_b = {
                    "gravida": "gravida",
                    "parity": "parity",
                    "gestational_age_weeks": "gestational age (weeks)",
                    "Age_yrs": "Age (yrs)",
                    "BMI": "BMI  [kg/m²]",
                    "diabetes": "diabetes",
                    "hypertension": "History of hypertension (y/n)",
                    "SystolicBP": "Systolic BP",
                    "DiastolicBP": "Diastolic BP",
                    "HB": "HB",
                    "fetal_weight": "fetal weight(kgs)",
                    "Protien_Uria": "Protien Uria",
                    "amniotic_fluid_levels": "amniotic fluid levels(cm)"
                }

                FEATURES_B = list(feature_map_b.values())
                x_b_dict = {feature_map_b.get(k, k): v for k, v in data.dict().items()}
                x_b = pd.DataFrame([[x_b_dict[col] for col in FEATURES_B]], columns=FEATURES_B)
                x_b_scaled = scaler_b.transform(x_b)
                disease_pred = int(model_b.predict(x_b_scaled)[0])
                disease_proba = model_b.predict_proba(x_b_scaled)[0]

                disease_map = {1: "low", 2: "mid", 0: "high"}
                disease_label = disease_map.get(disease_pred, "unknown")
                disease_prob = f"{disease_proba[disease_pred] * 100:.1f}%"

                result.update({
                    "Disease_Status": disease_label,
                    "Disease_Probability": disease_prob
                })
            else:
                # Use rule-based disease detection for high risk when Model B is not available
                risk_factors = 0
                
                # Check critical health indicators
                if data.diabetes == 1:
                    risk_factors += 2
                if data.hypertension == 1:
                    risk_factors += 2
                if data.Protien_Uria == 1:
                    risk_factors += 2
                if data.SystolicBP >= 140 or data.DiastolicBP >= 90:
                    risk_factors += 1
                if data.BMI >= 30:
                    risk_factors += 1
                if data.HB < 10:
                    risk_factors += 1
                if data.Age_yrs >= 35 or data.Age_yrs < 18:
                    risk_factors += 1
                if data.BS >= 140:
                    risk_factors += 1
                    
                # Determine disease status based on risk factors
                if risk_factors >= 5:
                    disease_label = "high"
                    disease_prob = f"{min(75 + risk_factors * 3, 95):.1f}%"
                elif risk_factors >= 3:
                    disease_label = "mid"
                    disease_prob = f"{45 + risk_factors * 5:.1f}%"
                else:
                    disease_label = "low"
                    disease_prob = f"{max(15 + risk_factors * 5, 20):.1f}%"
                
                result.update({
                    "Disease_Status": disease_label,
                    "Disease_Probability": disease_prob
                })
        else:
            # For low and mid risk, show N/A
            result.update({
                "Disease_Status": "N/A",
                "Disease_Probability": "0%"
            })

        # Generate advice using Groq model
        advice = generate_advice(result["Risk_Level"], result["Disease_Status"])
        result["AI_Advice"] = advice

        return result

    except Exception as e:
        return {"error": "Prediction failed", "details": str(e)}


# ===========================
# Send Alert Endpoint
# ===========================
@app.post("/send_alert/")
def send_alert(data: AlertData):
    try:
        sid = send_whatsapp_alert(data.Risk_Level, data.phone_number)
        if sid:
            return {"message": "Alert sent successfully!", "sid": sid}
        else:
            return {"message": "Alert not sent - Twilio not configured", "sid": None}
    except Exception as e:
        return {"error": "Failed to send alert", "details": str(e)}


# ===========================
# Run Server
# ===========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


         