import streamlit as st
import os

from src.parser import load_resumes
from src.retriever import ResumeRetriever
from src.ranking import ResumeRanker
from src.llm import generate_job_description

st.set_page_config(
    page_title="AI Resume Screening System",
    layout="wide"
)

st.title("🤖 AI Resume Screening System")

st.write("Generate a Job Description and shortlist the best candidates.")

# ------------------------------
# Job Details
# ------------------------------

st.header("Job Details")

role = st.text_input("Job Title")

location = st.text_input("Location")

experience = st.text_input("Experience Required")

employment_type = st.selectbox(
    "Employment Type",
    [
        "Full Time",
        "Part Time",
        "Internship",
        "Contract"
    ]
)

work_mode = st.selectbox(
    "Work Mode",
    [
        "Onsite",
        "Hybrid",
        "Remote"
    ]
)

salary = st.text_input("Salary")

notice_period = st.text_input("Notice Period")

education = st.text_input("Education")

technical_skills = st.text_area(
    "Technical Skills"
)

soft_skills = st.text_area(
    "Soft Skills"
)

responsibilities = st.text_area(
    "Responsibilities"
)

benefits = st.text_area(
    "Benefits"
)

# ------------------------------
# Generate JD
# ------------------------------

if st.button("Generate Job Description"):

    details = {

        "role": role,

        "location": location,

        "experience": experience,

        "employment_type": employment_type,

        "work_mode": work_mode,

        "salary": salary,

        "notice_period": notice_period,

        "education": education,

        "technical_skills": technical_skills,

        "soft_skills": soft_skills,

        "responsibilities": responsibilities,

        "benefits": benefits
    }

    with st.spinner("Generating Job Description..."):

        jd = generate_job_description(details)

    st.success("Job Description Generated")

    st.text_area(
        "Generated Job Description",
        jd,
        height=350
    )

    # ------------------------------
    # Load resumes
    # ------------------------------

    resumes = load_resumes("data/resumes")

    retriever = ResumeRetriever(resumes)

    results = retriever.retrieve(
        jd,
        top_k=min(5, len(resumes))
    )

    df = ResumeRanker.shortlist(results)

    ResumeRanker.save(df)

    st.header("Top Candidates")

    st.dataframe(df)

    st.success("Ranking Complete")