import os
import tempfile
import streamlit as st

from src.llm import generate_job_description
from src.resume_loader import (
    load_uploaded_resumes,
    extract_uploaded_file
)
from src.retriever import ResumeRetriever
from src.parser import ResumeRanker

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Screening System")

tab1, tab2 = st.tabs([
    "Generate Job Description",
    "Resume Screening"
])

##############################################################
# TAB 1
##############################################################

with tab1:

    st.header("Generate Job Description")

    col1, col2 = st.columns(2)

    with col1:

        role = st.text_input("Job Title")

        location = st.text_input("Location")

        experience = st.text_input("Experience")

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

    with col2:

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

        st.session_state["jd"] = jd

    if "jd" in st.session_state:

        st.success("Job Description Generated")

        st.text_area(
            "Generated Job Description",
            st.session_state["jd"],
            height=450
        )

        st.download_button(
            "Download JD",
            data=st.session_state["jd"],
            file_name="Job_Description.txt",
            mime="text/plain"
        )

##############################################################
# TAB 2
##############################################################

with tab2:

    st.header("Resume Screening")

    st.subheader("Step 1 : Upload Job Description")

    uploaded_jd = st.file_uploader(
        "Upload JD",
        type=[
            "pdf",
            "docx",
            "txt"
        ]
    )

    jd_text = ""

    if uploaded_jd is not None:

        jd_text = extract_uploaded_file(uploaded_jd)

    else:

        jd_text = st.text_area(
            "OR Paste Job Description",
            height=250
        )

    st.divider()

    st.subheader("Step 2 : Upload Resumes")

    uploaded_resumes = st.file_uploader(
        "Upload Multiple Resumes",
        type=[
            "pdf",
            "docx",
            "txt"
        ],
        accept_multiple_files=True
    )

    top_k = st.slider(
        "Number of Candidates",
        1,
        20,
        5
    )

    if st.button("Find Best Candidates"):

        if not jd_text.strip():

            st.error("Please upload or paste Job Description.")

        elif not uploaded_resumes:

            st.error("Please upload resumes.")

        else:

            with st.spinner("Reading resumes..."):

                resumes = load_uploaded_resumes(
                    uploaded_resumes
                )

            with st.spinner("Generating embeddings..."):

                retriever = ResumeRetriever(
                    resumes
                )

                results = retriever.retrieve(
                    jd_text,
                    top_k
                )

            df = ResumeRanker.shortlist(
                results,jd_text
            )

            os.makedirs(
                "outputs",
                exist_ok=True
            )

            excel_path = ResumeRanker.save(df)

            st.success("Ranking Complete")

            st.dataframe(
                df,
                use_container_width=True
            )

            with open(
                excel_path,
                "rb"
            ) as f:

                st.download_button(

                    "Download Excel",

                    f,

                    file_name="Shortlisted_Candidates.xlsx",

                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )