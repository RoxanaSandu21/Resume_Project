from sqlalchemy import Column, Integer, String, Text, Float, TIMESTAMP, ForeignKey, JSON
from sqlalchemy.sql import func
from database import Base

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())

class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255))
    email = Column(String(255))
    phone = Column(String(50))
    file_path = Column(String(512))
    parsed_text = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())

class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"))
    candidate_id = Column(Integer, ForeignKey("candidates.id", ondelete="CASCADE"))
    match_score = Column(Float)
    bias_report = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())