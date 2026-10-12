import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ATS & Bias Evaluation Dashboard", page_icon="📊", layout="wide")

st.title("📄 AI-Powered Recruitment System: ATS & Bias Evaluation")
st.write("Master's Dissertation Project — Automated Resume-Job Compatibility Scoring and Algorithmic Fairness Analysis.")

# Sidebar Navigation
menu = st.sidebar.selectbox("Navigation", ["Manage Jobs", "Upload CV & Candidate Leaderboard"])

if menu == "Manage Jobs":
    st.header("📌 Create a New Job Position")
    
    with st.form("job_form"):
        title = st.text_input("Job Title (e.g., Junior Java Developer)")
        description = st.text_area("Job Description (requirements, core technologies, responsibilities)")
        submit = st.form_submit_button("Create Job in System")
        
        if submit:
            if title and description:
                response = requests.post(f"{API_URL}/jobs/", json={"title": title, "description": description})
                if response.status_code == 200:
                    st.success("Job created successfully!")
                else:
                    st.error("An error occurred while creating the job.")
            else:
                st.warning("Please fill in all fields!")

    st.divider()
    st.subheader("📋 Active Job Postings")
    res = requests.get(f"{API_URL}/jobs/")
    if res.status_code == 200:
        jobs = res.json()
        if jobs:
            for job in jobs:
                with st.expander(f"🔹 {job['title']} (ID: {job['id']})"):
                    st.write(job['description'])
        else:
            st.info("No job postings available yet. Create one above.")

elif menu == "Upload CV & Candidate Leaderboard":
    st.header("📂 Candidate CV Upload & Compatibility Analysis")
    st.write("Select an active job position below, upload a candidate's PDF CV, and the system will automatically parse it, calculate the semantic match score, and evaluate potential bias.")
    
    res = requests.get(f"{API_URL}/jobs/")
    if res.status_code == 200:
        jobs = res.json()
        if not jobs:
            st.warning("No job positions found. Please create a job first from the 'Manage Jobs' menu!")
        else:
            job_dict = {f"{j['title']} (ID: {j['id']})": j['id'] for j in jobs}
            selected_job_label = st.selectbox("Select Target Job Position", list(job_dict.keys()))
            job_id = job_dict[selected_job_label]
            
            st.divider()
            uploaded_file = st.file_uploader("Upload Candidate CV (.PDF format)", type=["pdf"])
            
            if uploaded_file is not None:
                if st.button("Process CV & Evaluate Compatibility"):
                    with st.spinner("Extracting text, computing semantic embeddings, and generating bias report..."):
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        r = requests.post(f"{API_URL}/jobs/{job_id}/upload-cv/", files=files)
                        
                        if r.status_code == 200:
                            data = r.json()
                            st.success(f"CV for **{data['full_name']}** processed successfully!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(label="Match Score (Compatibility)", value=f"{data['match_score']}%")
                            with col2:
                                st.write("**Detected Bias Summary:**")
                                st.json(data['bias_report'])
                        else:
                            st.error(f"Processing error: {r.text}")
            
            st.divider()
            st.subheader("🏆 Dynamic Leaderboard & Fairness Report")
            st.write("Candidates are automatically ranked in real-time based on their semantic compatibility score.")
            
            lb_res = requests.get(f"{API_URL}/jobs/{job_id}/leaderboard/")
            if lb_res.status_code == 200:
                candidates = lb_res.json()
                if candidates:
                    for idx, cand in enumerate(candidates, 1):
                        score_color = "🟢" if cand['match_score'] > 70 else ("🟡" if cand['match_score'] > 40 else "🔴")
                        with st.expander(f"#{idx} - {cand['full_name']} {score_color} (Score: {cand['match_score']}%)"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**Email:** {cand['email']}")
                                st.write(f"**Phone:** {cand['phone']}")
                                st.write(f"**Submitted At:** {cand['submitted_at']}")
                            with col_b:
                                st.write("**Algorithmic Bias Evaluation:**")
                                st.json(cand['bias_report'])
                else:
                    st.info("No candidates registered for this job position yet.")