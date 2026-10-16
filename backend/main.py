import logging
import os
import uuid

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from database import engine, Base, get_db
import models
import schemas
from services.nlp_service import (
    extract_text_from_pdf,
    extract_cv_information,
    calculate_hybrid_match_score,
    evaluate_bias,
    extract_candidate_info,
)

# Reuses your existing SQLAlchemy models and schemas.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Resume-Job Compatibility Scoring and Bias Evaluation API", version="2.0")
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploaded_cvs"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health/")
def health_check():
    return {"status": "ok", "message": "API is running. Ollama is checked when NLP processing is requested."}


@app.post("/jobs/", response_model=schemas.JobResponse)
def create_job(job: schemas.JobCreate, db: Session = Depends(get_db)):
    if not job.title.strip() or not job.description.strip():
        raise HTTPException(status_code=422, detail="Job title and description are required.")

    db_job = models.Job(title=job.title.strip(), description=job.description.strip())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


@app.get("/jobs/", response_model=list[schemas.JobResponse])
def get_jobs(db: Session = Depends(get_db)):
    return db.query(models.Job).order_by(models.Job.id.desc()).all()


@app.post("/jobs/{job_id}/upload-cv/")
def upload_cv(job_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    original_filename = file.filename or "resume.pdf"
    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF CV files are accepted.")

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF is too large. Maximum allowed size is 10 MB.")

    # Use a generated server-side filename to avoid collisions and path traversal.
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.pdf")
    with open(file_path, "wb") as output_file:
        output_file.write(content)

    logger.info("Started resume processing for job_id=%s", job_id)
    try:
        logger.info("Extracting text from PDF for job_id=%s", job_id)
        cv_text = extract_text_from_pdf(file_path)
        logger.info("Extracted %s characters for job_id=%s", len(cv_text), job_id)
        logger.info("Extracting structured resume information for job_id=%s", job_id)
        cv_data = extract_cv_information(cv_text)
        candidate_info = extract_candidate_info(cv_text, cv_info=cv_data)
        logger.info("Scoring resume against job requirements for job_id=%s", job_id)
        match_result = calculate_hybrid_match_score(
            job.description,
            cv_text,
            job_title=job.title,
            cv_data=cv_data,
        )
        logger.info("Generating text-based bias screening for job_id=%s", job_id)
        bias_report = evaluate_bias(cv_text)
        logger.info("Finished resume scoring for job_id=%s with score=%.2f", job_id, match_result["score"])
    except ValueError as exc:
        logger.warning("Resume validation failed for job_id=%s: %s", job_id, exc)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("NLP service error for job_id=%s (%s)", job_id, type(exc).__name__)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected resume processing error for job_id=%s (%s)", job_id, type(exc).__name__)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Resume processing failed ({type(exc).__name__}). Check the backend terminal for details.") from exc

    bias_report["match_details"] = {
        "explanation": match_result["explanation"],
        "matched_skills": match_result["matched_skills"],
        "partial_requirements": match_result["partial_requirements"],
        "missing_skills": match_result["missing_skills"],
        "required_missing": match_result["required_missing"],
        "requirement_details": match_result["requirement_details"],
        "semantic_similarity": match_result["semantic_similarity"],
        "scoring_method": match_result["scoring_method"],
        "extracted_requirements_count": match_result["extracted_requirements_count"],
        "extracted_cv_skills_count": match_result["extracted_cv_skills_count"],
    }

    try:
        db_candidate = models.Candidate(
            full_name=candidate_info["full_name"],
            email=candidate_info["email"],
            phone=candidate_info["phone"],
            file_path=file_path,
            parsed_text=cv_text,
        )
        db.add(db_candidate)
        db.flush()  # Assign candidate ID before adding the evaluation.

        db_evaluation = models.Evaluation(
            job_id=job.id,
            candidate_id=db_candidate.id,
            match_score=match_result["score"],
            bias_report=bias_report,
        )
        db.add(db_evaluation)
        db.commit()
        db.refresh(db_candidate)
    except Exception as exc:
        db.rollback()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Database save failed: {exc}") from exc

    return {
        "message": "CV processed successfully.",
        "candidate_id": db_candidate.id,
        "full_name": db_candidate.full_name,
        "match_score": match_result["score"],
        "bias_report": bias_report,
        "extracted_information": {
            "skills": cv_data.get("skills", []),
            "experience": cv_data.get("experience", []),
            "education": cv_data.get("education", []),
            "certifications": cv_data.get("certifications", []),
            "languages": cv_data.get("languages", []),
        },
    }


@app.get("/jobs/{job_id}/leaderboard/")
def get_leaderboard(job_id: int, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    evaluations = (
        db.query(models.Evaluation, models.Candidate)
        .join(models.Candidate, models.Evaluation.candidate_id == models.Candidate.id)
        .filter(models.Evaluation.job_id == job_id)
        .order_by(models.Evaluation.match_score.desc())
        .all()
    )

    results = []
    for evaluation, candidate in evaluations:
        results.append({
            "candidate_id": candidate.id,
            "full_name": candidate.full_name,
            "email": candidate.email,
            "phone": candidate.phone,
            "match_score": evaluation.match_score,
            "bias_report": evaluation.bias_report,
            "submitted_at": evaluation.created_at,
        })
    return results
