from fastapi import FastAPI, UploadFile, File
import joblib
from pydantic import BaseModel
import fitz
import json
from google import genai  # SAHI ✅
import os

app = FastAPI()
model = joblib.load("model.pkl")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class Student(BaseModel):
    Cgpa : float
    interships:int
    projects:int

class prediction_response(BaseModel):
    placement_probility : float

@app.post("/predict",response_model =prediction_response )
def predcit(student:Student):
    prediction = model.predict([[student.Cgpa,student.interships,student.projects]])
    return {"placement_probility":round(float(prediction[0]),2)}

def extract_text_from_pdf(file_bytes :bytes) ->str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text =""
    for page in doc:
        text += page.get_text()
        return text.strip()
    

def analyze_with_gemini(resume_text: str) -> dict:
    """Gemini se full analysis lo"""
    
    prompt = f"""
    Tujhe ek resume analyzer ka kaam karna hai. Neeche ek resume ka text hai.
    
    Ye kaam karo aur SIRF valid JSON return karo, koi explanation nahi:
    
    {{
        "name": "candidate ka naam (agar mile)",
        "score": <0-100 ka integer, overall resume quality score>,
        "score_breakdown": {{
            "skills": <0-25>,
            "experience": <0-25>,
            "education": <0-25>,
            "formatting_and_clarity": <0-25>
        }},
        "extracted_skills": ["skill1", "skill2", ...],
        "strengths": ["strength 1", "strength 2", "strength 3"],
        "improvements": [
            {{
                "issue": "kya problem hai",
                "suggestion": "kya kare"
            }}
        ],
        "ats_keywords_missing": ["keyword1", "keyword2"],
        "one_line_verdict": "ek line mein honest feedback"
    }}
    
    Resume text:
    {resume_text[:4000]}
    """
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
    raw = response.text.strip()
    
    # JSON clean karo (Gemini kabhi kabhi ```json wrap karta hai)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    return json.loads(raw)


@app.post("/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    
    # Sirf PDF allow karo
    if file.content_type != "application/pdf":
        return {"error": "Sirf PDF file upload karo!"}
    
    # File bytes read karo
    file_bytes = await file.read()
    
    # Text extract karo
    resume_text = extract_text_from_pdf(file_bytes)
    
    if not resume_text or len(resume_text) < 50:
        return {"error": "PDF se text nahi nikal paya. Scanned image PDF hai kya?"}
    
    # Gemini se analyze karo
    analysis = analyze_with_gemini(resume_text)
    
    return analysis
