import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts raw text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def calculate_match_score(job_description: str, cv_text: str) -> float:
    """Calculates semantic similarity between the job description and the CV."""
    if not cv_text.strip():
        return 0.0
    
    emb_job = model.encode(job_description, convert_to_tensor=True)
    emb_cv = model.encode(cv_text, convert_to_tensor=True)
    
    similarity = util.cos_sim(emb_job, emb_cv)
    score = float(similarity.item()) * 100  # convert to percentage (0 - 100)
    
    return round(score, 2)

def evaluate_bias(cv_text: str) -> dict:
    """Analyzes the CV text to detect potential bias factors."""
    text_lower = cv_text.lower()
    
    has_photo_mention = bool(re.search(r'\b(photo|picture|fotografie|poza)\b', text_lower))
    has_marital_status = bool(re.search(r'\b(married|single|children|family|casatorit|necasatorit|copii)\b', text_lower))
    
    risk_factors = 0
    warnings = []
    
    if has_photo_mention:
        risk_factors += 1
        warnings.append("CV contains photo references (risk of visual/demographic bias).")
        
    if has_marital_status:
        risk_factors += 1
        warnings.append("CV contains marital status or family details.")

    bias_level = "Low" if risk_factors == 0 else ("Medium" if risk_factors == 1 else "High")

    return {
        "bias_level": bias_level,
        "risk_score_factors": risk_factors,
        "warnings": warnings,
        "recommendation": "Anonymization of personal data is recommended before interview selection." if risk_factors > 0 else "CV is properly structured from a fairness perspective."
    }

def extract_candidate_info(cv_text: str) -> dict:
    """Extracts basic info (email, phone, name estimation) from the CV using regex."""
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', cv_text)
    phone_match = re.search(r'(?:\+40|0040|0)?\s*([732][0-9]{8})\b', cv_text)
    
    email = email_match.group(0) if email_match else "Unspecified"
    phone = phone_match.group(0) if phone_match else "Unspecified"
    
    lines = [line.strip() for line in cv_text.split('\n') if line.strip()]
    full_name = lines[0] if lines else "Unknown Candidate"

    return {
        "full_name": full_name,
        "email": email,
        "phone": phone
    }