import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Resume-Job Compatibility & Bias Evaluation",
    page_icon="📄",
    layout="wide",
)

st.title("Resume-Job Compatibility Scoring and Bias Evaluation")
st.caption("Master's dissertation prototype — AI-assisted extraction, evidence-based matching and textual proxy screening.")

menu = st.sidebar.selectbox("Navigation", ["Manage Jobs", "Upload CV & Candidate Leaderboard"])


def api_error(response):
    try:
        payload = response.json()
        return payload.get("detail", response.text)
    except Exception:
        return response.text


def show_requirement_details(details):
    if not details:
        st.info("No per-requirement details returned.")
        return

    for index, item in enumerate(details, start=1):
        status = item.get("status", "unknown")
        if status == "matched":
            marker = "✅"
        elif status == "partial":
            marker = "🟡"
        else:
            marker = "❌"

        title = f"{marker} {index}. {item.get('requirement', 'Requirement')} — {item.get('score', 0)}%"
        with st.expander(title):
            st.write(f"**Priority:** {item.get('priority', 'unspecified')}")
            st.write(f"**Category:** {item.get('category', 'other')}")
            st.write(f"**Status:** {status}")
            st.write(f"**Original job text:** {item.get('source_text', '')}")
            st.write(f"**Explanation:** {item.get('explanation', '')}")
            evidence = item.get("evidence", [])
            if evidence:
                st.write("**CV evidence:**")
                for entry in evidence:
                    st.markdown(f"- **{entry.get('label', 'Evidence')}**: “{entry.get('quote', '')}”")
            else:
                st.write("**CV evidence:** No supporting excerpt selected.")
            st.caption(f"Top semantic retrieval similarity: {item.get('semantic_similarity', 0)}% (not a probability).")


if menu == "Manage Jobs":
    st.header("Create a job position")
    with st.form("job_form"):
        title = st.text_input("Job title", placeholder="e.g. Junior Java Developer")
        description = st.text_area(
            "Full job description",
            height=260,
            placeholder="Paste the complete job posting, including requirements and responsibilities.",
        )
        submit = st.form_submit_button("Create job")

    if submit:
        if not title.strip() or not description.strip():
            st.warning("Please enter both a title and a description.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/jobs/",
                    json={"title": title.strip(), "description": description.strip()},
                    timeout=30,
                )
                if response.status_code in (200, 201):
                    st.success("Job created successfully.")
                else:
                    st.error(f"Could not create job: {api_error(response)}")
            except requests.RequestException as exc:
                st.error(f"Could not connect to the API: {exc}")

    st.divider()
    st.subheader("Saved job positions")
    try:
        response = requests.get(f"{API_URL}/jobs/", timeout=20)
        response.raise_for_status()
        jobs = response.json()
        if not jobs:
            st.info("No jobs yet. Create one above.")
        for job in jobs:
            with st.expander(f"{job['title']} (ID: {job['id']})"):
                st.write(job["description"])
    except requests.RequestException as exc:
        st.error(f"Could not connect to the API: {exc}")

else:
    st.header("Upload a CV and evaluate compatibility")
    try:
        response = requests.get(f"{API_URL}/jobs/", timeout=20)
        response.raise_for_status()
        jobs = response.json()
    except requests.RequestException as exc:
        st.error(f"Could not connect to the API: {exc}")
        jobs = []

    if not jobs:
        st.info("Create a job position first in Manage Jobs.")
    else:
        labels = {f"{job['title']} (ID: {job['id']})": job["id"] for job in jobs}
        selected_label = st.selectbox("Target job", list(labels.keys()))
        job_id = labels[selected_label]
        uploaded_file = st.file_uploader("Upload CV (PDF, maximum 10 MB)", type=["pdf"])

        if uploaded_file is not None and st.button("Process CV and evaluate", type="primary"):
            with st.spinner("Extracting structured information and comparing evidence locally..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(
                        f"{API_URL}/jobs/{job_id}/upload-cv/",
                        files=files,
                        timeout=900,
                    )
                except requests.RequestException as exc:
                    st.error(f"Could not connect to the API: {exc}")
                else:
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Processed CV for {data.get('full_name', 'Unknown Candidate')}.")
                        report = data.get("bias_report", {})
                        match = report.get("match_details", {})

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Compatibility score", f"{data.get('match_score', 0)}%")
                        col2.metric("Requirements matched", len(match.get("matched_skills", [])))
                        col3.metric("Requirements missing", len(match.get("missing_skills", [])))

                        st.subheader("How the score was calculated")
                        st.write(match.get("explanation", "No explanation returned."))
                        st.caption(f"Scoring method: {match.get('scoring_method', 'unknown')}")
                        st.caption(
                            f"Mean top semantic retrieval similarity: {match.get('semantic_similarity', 0)}%. "
                            "This is a retrieval diagnostic, not the final compatibility score."
                        )

                        if match.get("required_missing"):
                            st.warning("Required criteria without sufficient CV evidence:")
                            for requirement in match["required_missing"]:
                                st.markdown(f"- {requirement}")

                        st.subheader("Requirement-by-requirement evaluation")
                        show_requirement_details(match.get("requirement_details", []))

                        with st.expander("Extracted candidate information"):
                            st.json(data.get("extracted_information", {}))

                        st.subheader("Textual proxy-indicator screening")
                        st.write(f"**Heuristic indicator level:** {report.get('bias_level', 'Unknown')}")
                        for warning in report.get("warnings", []):
                            st.warning(warning)
                        if not report.get("warnings"):
                            st.info("No configured textual indicators were detected; this does not prove the system is fair.")
                        st.caption(report.get("scope_note", ""))
                        st.write(report.get("recommendation", ""))
                    else:
                        st.error(f"Processing error ({response.status_code}): {api_error(response)}")

        st.divider()
        st.subheader("Candidate leaderboard")
        try:
            response = requests.get(f"{API_URL}/jobs/{job_id}/leaderboard/", timeout=30)
            response.raise_for_status()
            candidates = response.json()
            if not candidates:
                st.info("No candidates have been processed for this job yet.")
            else:
                for rank, candidate in enumerate(candidates, start=1):
                    score = candidate.get("match_score", 0)
                    with st.expander(f"#{rank} — {candidate.get('full_name', 'Unknown')} — {score}%"):
                        left, right = st.columns(2)
                        with left:
                            st.write(f"**Email:** {candidate.get('email', 'Unspecified')}")
                            st.write(f"**Phone:** {candidate.get('phone', 'Unspecified')}")
                            st.write(f"**Submitted:** {candidate.get('submitted_at', 'Unknown')}")
                        with right:
                            report = candidate.get("bias_report", {}) or {}
                            st.write(f"**Textual indicator level:** {report.get('bias_level', 'Unknown')}")
                            st.write(report.get("scope_note", ""))
                        match = (candidate.get("bias_report", {}) or {}).get("match_details", {})
                        show_requirement_details(match.get("requirement_details", []))
        except requests.RequestException as exc:
            st.error(f"Could not load the leaderboard: {exc}")
