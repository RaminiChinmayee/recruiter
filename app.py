import streamlit as st
import pandas as pd


from src.llm import generate_job_description


from src.resume_loader import (
    load_multiple_resumes,
    extract_uploaded_file
)


from src.retriever import HybridRetriever


from src.reranker import ResumeReranker


from src.parser import parse_resume


from src.report import export_excel



# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(

    page_title="AI Resume Screener",

    page_icon="🚀",

    layout="wide"

)



st.title(
    "🚀 AI Resume Screening System"
)


st.caption(
    "GenAI + RAG + FAISS + CrossEncoder + ATS Ranking"
)



tab1, tab2 = st.tabs(

    [

        "📝 Generate Job Description",

        "📄 Resume Screening"

    ]

)



# ======================================================
# TAB 1 : JOB DESCRIPTION GENERATOR
# ======================================================


with tab1:


    st.header(
        "Generate ATS Optimized Job Description"
    )



    col1, col2 = st.columns(2)



    with col1:


        company_name = st.text_input(

            "Company Name",

            "Microsoft"

        )


        role = st.text_input(

            "Job Role",

            "Machine Learning Engineer"

        )


        experience = st.text_input(

            "Experience",

            "2-4 Years"

        )


        location = st.text_input(

            "Location",

            "Hyderabad"

        )


        skills = st.text_area(

            "Technical Skills",

            "Python, SQL, Machine Learning, TensorFlow, AWS"

        )



    with col2:


        education = st.text_input(

            "Education",

            "B.Tech Computer Science"

        )


        responsibilities = st.text_area(

            "Responsibilities",

            """
Build ML models
Deploy APIs
Analyze data
Improve model performance
"""

        )


        softskills = st.text_area(

            "Soft Skills",

            "Communication, teamwork, problem solving"

        )


        workmode = st.selectbox(

            "Work Mode",

            [

                "Remote",

                "Hybrid",

                "Onsite"

            ]

        )


        salary = st.text_input(

            "Salary",

            "8-12 LPA"

        )


        benefits = st.text_area(

            "Benefits",

            "Insurance, PF, Flexible Hours"

        )



    if st.button(
        "✨ Generate JD"
    ):


        details = {


            "company_name":
            company_name,


            "role":
            role,


            "experience":
            experience,


            "location":
            location,


            "technical_skills":
            skills,


            "education":
            education,


            "responsibilities":
            responsibilities,


            "soft_skills":
            softskills,


            "work_mode":
            workmode,


            "employment_type":
            "Full Time",


            "salary":
            salary,


            "benefits":
            benefits

        }



        with st.spinner(

            "Generating JD using Llama 3.3..."

        ):


            jd = generate_job_description(

                details

            )


            st.session_state["jd"] = jd



    if "jd" in st.session_state:


        st.subheader(

            "Generated Job Description"

        )


        st.text_area(

            "Preview",

            st.session_state["jd"],

            height=500

        )


        st.download_button(

            "⬇ Download JD",

            data=st.session_state["jd"],

            file_name="job_description.txt",

            mime="text/plain"

        )





# ======================================================
# TAB 2 : RESUME SCREENING
# ======================================================


with tab2:


    st.header(

        "AI Candidate Screening"

    )



    uploaded_jd = st.file_uploader(

        "Upload Job Description",

        type=[

            "pdf",

            "docx",

            "txt"

        ]

    )



    resumes = st.file_uploader(

        "Upload Candidate Resumes",

        type=[

            "pdf",

            "docx",

            "txt"

        ],

        accept_multiple_files=True

    )



    if st.button(

        "🚀 Analyze Candidates"

    ):



        if uploaded_jd is None or not resumes:


            st.warning(

                "Upload JD and resumes"

            )


            st.stop()



        # --------------------------
        # Read Documents
        # --------------------------


        with st.spinner(

            "Reading documents..."

        ):



            jd_text = extract_uploaded_file(

                uploaded_jd

            )



            resume_data = load_multiple_resumes(

                resumes

            )



        if not resume_data:


            st.error(

                "No resumes extracted"

            )

            st.stop()



        # --------------------------
        # Retrieval
        # --------------------------


        with st.spinner(

            "Searching candidates..."

        ):


            retriever = HybridRetriever(

                resume_data

            )


            candidates = retriever.search(

                jd_text,

                top_k=20

            )



        if not candidates:


            st.error(

                "No candidates found"

            )

            st.stop()



        # --------------------------
        # Reranking
        # --------------------------


        with st.spinner(

            "Reranking candidates..."

        ):



            reranker = ResumeReranker()



            ranked = reranker.rerank(

                jd_text,

                candidates,

                top_k=10

            )



        # --------------------------
        # Parse Resumes
        # --------------------------


        results = []



        with st.spinner(

            "Parsing resumes..."

        ):


            for candidate in ranked:



                try:


                    resume_json = parse_resume(

                        candidate["text"]

                    )


                except Exception as e:


                    st.warning(

                        f"Parsing failed {candidate['name']}"

                    )


                    resume_json = {}



                results.append({


                    "Candidate":

                    resume_json.get(

                        "name",

                        candidate["name"]

                    ),



                    "Email":

                    resume_json.get(

                        "email",

                        ""

                    ),



                    "Phone":

                    resume_json.get(

                        "phone",

                        ""

                    ),



                    "Location":

                    resume_json.get(

                        "location",

                        ""

                    ),



                    "Skills":

                    ", ".join(

                        resume_json.get(

                            "skills",

                            []

                        )

                    ),



                    "Experience":

                    resume_json.get(

                        "experience",

                        ""

                    ),



                    "Similarity Score":

                    round(

                        candidate["score"],

                        3

                    )

                })



        # --------------------------
        # Display Results
        # --------------------------


        df = pd.DataFrame(results)



        st.success(

            "Candidate screening completed"

        )



        st.subheader(

            "🏆 Ranked Candidates"

        )



        st.dataframe(

            df,

            use_container_width=True

        )



        # --------------------------
        # Analytics
        # --------------------------


        col1,col2,col3 = st.columns(3)



        with col1:


            st.metric(

                "Total Resumes",

                len(df)

            )



        with col2:


            st.metric(

                "Top Candidate",

                df.iloc[0]["Candidate"]

                if len(df)>0

                else "-"

            )



        with col3:


            st.metric(

                "Average Score",

                round(

                    df["Similarity Score"].mean(),

                    3

                )

                if len(df)>0

                else 0

            )



        # --------------------------
        # Excel Export
        # --------------------------


        excel_path = export_excel(

            df

        )


        with open(

            excel_path,

            "rb"

        ) as file:



            st.download_button(

                "⬇ Download Excel Report",

                data=file,

                file_name="candidate_report.xlsx",

                mime=
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            )