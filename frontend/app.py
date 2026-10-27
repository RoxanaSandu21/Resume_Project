from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
CONNECT_TIMEOUT = 4
NORMAL_TIMEOUT = (5, 30)
PROCESSING_TIMEOUT = (10, 1800)  # The local language model can take several minutes.

st.set_page_config(
    page_title="Resume-Job Compatibility and Bias Evaluation",
    page_icon="📄",
    layout="wide",
)

st.title("Resume-Job Compatibility Scoring and Bias Evaluation")
st.caption(
    "Master's dissertation prototype: evidence-based resume matching and "
    "algorithmic fairness screening."
)


def response_detail(response: requests.Response) -> str:
    """Return a useful API error message from a JSON or text response."""
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return str(detail)

    return response.text.strip() or f"HTTP {response.status_code}"


def show_backend_startup_help() -> None:
    st.error(f"Cannot connect to the FastAPI backend at `{API_URL}`.")
    st.write(
        "The Streamlit frontend and FastAPI backend run as separate processes. "
        "Keep the backend running in a second terminal."
    )
    st.markdown("From the project root, start the backend with:")
    st.code(
        "cd backend\n"
        ".\\venv\\Scripts\\Activate.ps1\n"
        "uvicorn main:app --reload --host 127.0.0.1 --port 8000",
        language="powershell",
    )
    st.markdown("Then start the frontend in another terminal:")
    st.code(
        "cd frontend\n"
        "..\\backend\\venv\\Scripts\\Activate.ps1\n"
        "streamlit run app.py",
        language="powershell",
    )
    st.caption(
        "If your virtual environment is stored somewhere else, activate that "
        "environment instead. The backend also needs a valid DATABASE_URL and "
        "a running Ollama service for CV processing."
    )


def api_request(
    method: str,
    path: str,
    *,
    timeout: tuple[int, int] | int = NORMAL_TIMEOUT,
    **kwargs: Any,
) -> requests.Response | None:
    """Make an API request and display readable errors instead of a traceback."""
    try:
        response = requests.request(
            method,
            f"{API_URL}{path}",
            timeout=timeout,
            **kwargs,
        )
        return response
    except requests.exceptions.ConnectionError:
        show_backend_startup_help()
        st.stop()
    except requests.exceptions.Timeout:
        st.error(
            "The API request timed out. Check the backend terminal for errors. "
            "If this happened during CV processing, also confirm that Ollama is "
            "running and that the configured model is available."
        )
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None


def fetch_jobs() -> list[dict[str, Any]] | None:
    response = api_request("GET", "/jobs/")
    if response is None:
        return None
    if not response.ok:
        st.error(
            f"The backend could not load job postings "
            f"(HTTP {response.status_code}): {response_detail(response)}"
        )
        return None
    try:
        payload = response.json()
    except ValueError:
        st.error("The backend returned an invalid response while loading job postings.")
        return None
    return payload if isinstance(payload, list) else []


# Fail early with actionable guidance if the backend process is not running.
try:
    health_response = requests.get(f"{API_URL}/health/", timeout=CONNECT_TIMEOUT)
except requests.exceptions.RequestException:
    show_backend_startup_help()
    st.stop()

if not health_response.ok:
    st.error(
        f"The backend health check failed (HTTP {health_response.status_code}). "
        "Check the backend terminal for the startup error."
    )
    st.stop()

menu = st.sidebar.radio(
    "Navigation",
    ["Manage Jobs", "Upload CV and Candidate Leaderboard"],
)

if menu == "Manage Jobs":
    st.header("Create a Job Posting")

    with st.form("job_form"):
        title = st.text_input("Job title", placeholder="Junior Java Developer")
        description = st.text_area(
            "Job description",
            placeholder="Paste the full job description, including requirements and responsibilities.",
            height=220,
        )
        submit_job = st.form_submit_button("Create Job")

    if submit_job:
        if not title.strip() or not description.strip():
            st.warning("Enter both a job title and a job description.")
        else:
            response = api_request(
                "POST",
                "/jobs/",
                json={"title": title.strip(), "description": description.strip()},
            )
            if response is not None:
                if response.ok:
                    st.success("Job posting created successfully.")
                else:
                    st.error(
                        f"Could not create the job (HTTP {response.status_code}): "
                        f"{response_detail(response)}"
                    )

    st.divider()
    st.subheader("Existing Job Postings")
    jobs = fetch_jobs()
    if jobs is not None:
        if not jobs:
            st.info("No job postings have been created yet.")
        else:
            for job in jobs:
                with st.expander(f"{job.get('title', 'Untitled job')} (ID: {job.get('id')})"):
                    st.write(job.get("description", ""))

else:
    st.header("Resume Upload and Compatibility Analysis")
    st.write(
        "Choose a job posting, upload a PDF resume, and run the compatibility "
        "analysis. Processing may take several minutes the first time while the "
        "local models are loading. Before uploading, you can test generation at "
        "`http://127.0.0.1:8000/diagnostics/ollama-generation/`. "
        "During processing, the timer below shows elapsed time; the backend terminal "
        "shows which NLP stage is running."
    )

    jobs = fetch_jobs()
    if jobs is None:
        st.stop()
    if not jobs:
        st.info("No job postings are available. Create a job under **Manage Jobs** first.")
        st.stop()

    jobs_by_label = {
        f"{job.get('title', 'Untitled job')} (ID: {job['id']})": job["id"]
        for job in jobs
    }
    selected_label = st.selectbox("Target job", list(jobs_by_label.keys()))
    selected_job_id = jobs_by_label[selected_label]

    uploaded_file = st.file_uploader("Upload a resume in PDF format", type=["pdf"])

    if uploaded_file is not None and st.button("Process Resume", type="primary"):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        }
        response = None
        started = time.perf_counter()
        with st.status(
            "Starting resume analysis...",
            expanded=True,
        ) as processing_status:
            st.write(
                "The browser waits for the backend to finish. Watch the backend "
                "terminal for detailed model timings. The first run may also "
                "download and initialize the embedding model."
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    requests.post,
                    f"{API_URL}/jobs/{selected_job_id}/upload-cv/",
                    files=files,
                    timeout=PROCESSING_TIMEOUT,
                )
                while not future.done():
                    elapsed = int(time.perf_counter() - started)
                    processing_status.update(
                        label=(
                            f"Processing resume — {elapsed} seconds elapsed. "
                            "See the backend terminal for the current stage."
                        ),
                        state="running",
                        expanded=True,
                    )
                    time.sleep(1)

                try:
                    response = future.result()
                    elapsed = round(time.perf_counter() - started, 1)
                    processing_status.update(
                        label=f"Backend response received after {elapsed} seconds.",
                        state="complete" if response.ok else "error",
                        expanded=True,
                    )
                except requests.exceptions.Timeout:
                    elapsed = round(time.perf_counter() - started, 1)
                    processing_status.update(
                        label=f"Request timed out after {elapsed} seconds.",
                        state="error",
                        expanded=True,
                    )
                    st.error(
                        "The frontend timed out while waiting for the backend. "
                        "Check the backend terminal: an Ollama task may still be running "
                        "or may have failed. Try the Ollama diagnostic URL shown above."
                    )
                except requests.exceptions.RequestException as exc:
                    processing_status.update(
                        label="Could not reach the backend.",
                        state="error",
                        expanded=True,
                    )
                    st.error(f"Resume-processing request failed: {exc}")

        if response is not None:
            if response.ok:
                try:
                    result = response.json()
                except ValueError:
                    st.error("The backend returned an invalid response after processing the resume.")
                else:
                    st.success(
                        f"Resume processed successfully for "
                        f"**{result.get('full_name', 'Unknown Candidate')}**."
                    )
                    left, right = st.columns(2)
                    with left:
                        st.metric(
                            "Compatibility score",
                            f"{result.get('match_score', 0)}%",
                        )
                        report = result.get("bias_report") or {}
                        match_details = report.get("match_details") or {}
                        st.subheader("Match Explanation")
                        st.write(match_details.get("explanation", "No explanation was returned."))

                        st.subheader("Matched Requirements")
                        matched = match_details.get("matched_skills", [])
                        st.write(matched if matched else "No fully matched requirements were identified.")

                        st.subheader("Partially Matched Requirements")
                        partial = match_details.get("partial_requirements", [])
                        st.write(partial if partial else "No partial matches were identified.")

                        st.subheader("Requirements Without Sufficient Evidence")
                        missing = match_details.get("missing_skills", [])
                        st.write(missing if missing else "No missing requirements were identified.")

                        if match_details.get("required_missing"):
                            st.warning(
                                "Required criteria without sufficient evidence: "
                                + ", ".join(match_details["required_missing"])
                            )

                        if match_details.get("requirement_details"):
                            with st.expander("View requirement-by-requirement evidence"):
                                st.json(match_details["requirement_details"])

                    with right:
                        st.subheader("Bias Screening")
                        st.json(report)
                    if result.get("extracted_information"):
                        with st.expander("Extracted Resume Information"):
                            st.json(result["extracted_information"])
            else:
                st.error(
                    f"Resume processing failed (HTTP {response.status_code}): "
                    f"{response_detail(response)}"
                )

    st.divider()
    st.subheader("Candidate Leaderboard")
    leaderboard_response = api_request(
        "GET",
        f"/jobs/{selected_job_id}/leaderboard/",
    )
    if leaderboard_response is not None:
        if leaderboard_response.ok:
            try:
                candidates = leaderboard_response.json()
            except ValueError:
                st.error("The backend returned an invalid leaderboard response.")
                candidates = []
            if candidates:
                for rank, candidate in enumerate(candidates, start=1):
                    score = float(candidate.get("match_score") or 0)
                    with st.expander(
                        f"#{rank} — {candidate.get('full_name', 'Unknown Candidate')} "
                        f"(Score: {score:.2f}%)"
                    ):
                        details_left, details_right = st.columns(2)
                        with details_left:
                            st.write(f"**Email:** {candidate.get('email') or 'Unspecified'}")
                            st.write(f"**Phone:** {candidate.get('phone') or 'Unspecified'}")
                            st.write(f"**Submitted:** {candidate.get('submitted_at') or 'Unknown'}")
                            match_details = (candidate.get("bias_report") or {}).get("match_details", {})
                            st.write("**Matched requirements:**", match_details.get("matched_skills", []))
                            st.write("**Partial requirements:**", match_details.get("partial_requirements", []))
                            st.write("**Missing requirements:**", match_details.get("missing_skills", []))
                        with details_right:
                            st.write("**Bias screening report**")
                            st.json(candidate.get("bias_report") or {})
            else:
                st.info("No resumes have been evaluated for this job yet.")
        else:
            st.error(
                f"Could not load the leaderboard (HTTP {leaderboard_response.status_code}): "
                f"{response_detail(leaderboard_response)}"
            )
