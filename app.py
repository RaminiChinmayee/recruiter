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

from src.scorer import ATSScorer

from src.evaluator import evaluate_candidate

from src.report import export_excel

from src.dashboard import (
    ats_distribution,
    top_skills,
    experience_chart
)

from src.config import settings

# ======================================================
# EVALUATION METRICS
# ======================================================

from src.evaluation_metrics import (
    precision_at_k,
    recall_at_k,
    mrr_at_k,
    candidate_ndcg_at_k,
    spearman_correlation,
    score_mae
)


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🚀",
    layout="wide"
)


# ======================================================
# TITLE
# ======================================================

st.title("🚀 AI Resume Screening System")

st.caption(
    "GenAI + RAG + FAISS + BM25 + CrossEncoder + ATS Ranking"
)


# ======================================================
# SESSION STATE
# ======================================================

if "jd" not in st.session_state:
    st.session_state["jd"] = None

if "screening_results" not in st.session_state:
    st.session_state["screening_results"] = None


# ======================================================
# TABS
# ======================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📝 Generate Job Description",
        "📄 Resume Screening",
        "📊 Evaluation"
    ]
)


# ======================================================
# TAB 1
# JOB DESCRIPTION GENERATOR
# ======================================================

with tab1:

    st.header(
        "Generate ATS Optimized Job Description"
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------
    # LEFT
    # --------------------------------------------------

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

    # --------------------------------------------------
    # RIGHT
    # --------------------------------------------------

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

    # --------------------------------------------------
    # GENERATE JD
    # --------------------------------------------------

    if st.button("✨ Generate JD"):

        details = {

            "company_name": company_name,

            "role": role,

            "experience": experience,

            "location": location,

            "technical_skills": skills,

            "education": education,

            "responsibilities": responsibilities,

            "soft_skills": softskills,

            "work_mode": workmode,

            "employment_type": "Full Time",

            "salary": salary,

            "benefits": benefits
        }

        with st.spinner(
            "Generating Job Description..."
        ):

            try:

                jd = generate_job_description(
                    details
                )

                st.session_state["jd"] = jd

                st.success(
                    "Job Description generated successfully."
                )

            except Exception as e:

                st.error(
                    f"JD generation failed: {e}"
                )

    # --------------------------------------------------
    # DISPLAY JD
    # --------------------------------------------------

    if st.session_state["jd"]:

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
# TAB 2
# RESUME SCREENING
# ======================================================

with tab2:

    st.header(
        "AI Candidate Screening"
    )

    # --------------------------------------------------
    # UPLOAD JD
    # --------------------------------------------------

    uploaded_jd = st.file_uploader(
        "Upload Job Description",
        type=[
            "pdf",
            "docx",
            "txt"
        ],
        key="jd_upload"
    )

    # --------------------------------------------------
    # UPLOAD RESUMES
    # --------------------------------------------------

    resumes = st.file_uploader(
        "Upload Candidate Resumes",
        type=[
            "pdf",
            "docx",
            "txt"
        ],
        accept_multiple_files=True,
        key="resume_upload"
    )

    # --------------------------------------------------
    # ANALYZE
    # --------------------------------------------------

    analyze = st.button(
        "🚀 Analyze Candidates"
    )

    if analyze:

        # ==================================================
        # VALIDATION
        # ==================================================

        if uploaded_jd is None:

            st.warning(
                "Please upload a Job Description."
            )

            st.stop()

        if not resumes:

            st.warning(
                "Please upload at least one resume."
            )

            st.stop()

        # ==================================================
        # PROGRESS
        # ==================================================

        progress = st.progress(0)

        status = st.empty()

        # ==================================================
        # STAGE 1
        # DOCUMENT EXTRACTION
        # ==================================================

        status.info(
            "📖 Reading Job Description and resumes..."
        )

        try:

            jd_text = extract_uploaded_file(
                uploaded_jd
            )

            resume_data = load_multiple_resumes(
                resumes
            )

        except Exception as e:

            st.error(
                f"Document extraction failed: {e}"
            )

            st.stop()

        if not jd_text or not jd_text.strip():

            st.error(
                "The Job Description could not be extracted."
            )

            st.stop()

        if not resume_data:

            st.error(
                "No valid resumes could be extracted."
            )

            st.stop()

        progress.progress(20)

        status.success(
            f"✅ Extracted {len(resume_data)} resume(s)"
        )

        # ==================================================
        # STAGE 2
        # HYBRID RETRIEVAL
        # ==================================================

        status.info(
            "🔍 Building semantic + keyword search..."
        )

        try:

            retriever = HybridRetriever(
                resume_data
            )

            candidates = retriever.search(
                jd_text,
                top_k=settings.RETRIEVAL_TOP_K
            )

        except Exception as e:

            st.error(
                f"Candidate retrieval failed: {e}"
            )

            st.stop()

        if not candidates:

            st.error(
                "No candidates found."
            )

            st.stop()

        progress.progress(40)

        status.success(
            f"✅ Retrieved {len(candidates)} candidate(s)"
        )

        # ==================================================
        # STAGE 3
        # RERANKING
        # ==================================================

        status.info(
            "🎯 Reranking candidates using CrossEncoder..."
        )

        try:

            reranker = ResumeReranker()

            ranked = reranker.rerank(
                jd_text,
                candidates,
                top_k=settings.RERANK_TOP_K
            )

        except Exception as e:

            st.error(
                f"Reranking failed: {e}"
            )

            st.stop()

        if not ranked:

            st.error(
                "Reranking returned no candidates."
            )

            st.stop()

        progress.progress(60)

        status.success(
            f"✅ Reranked {len(ranked)} candidate(s)"
        )

        # ==================================================
        # STAGE 4
        # PARSE RESUMES + ATS
        # ==================================================

        status.info(
            "📄 Extracting candidate information..."
        )

        scorer = ATSScorer()

        results = []

        for position, candidate in enumerate(ranked):

            # --------------------------------------------------
            # PARSE
            # --------------------------------------------------

            try:

                resume_json = parse_resume(
                    candidate["text"]
                )

            except Exception as e:

                st.warning(
                    f"Parsing failed for "
                    f"{candidate.get('name', 'candidate')}: {e}"
                )

                resume_json = {

                    "name": candidate.get(
                        "name",
                        ""
                    ),

                    "email": "",

                    "phone": "",

                    "location": "",

                    "skills": [],

                    "education": [],

                    "experience": "",

                    "projects": [],

                    "certifications": []
                }

            # --------------------------------------------------
            # CANDIDATE NAME
            # --------------------------------------------------

            candidate_name = (

                resume_json.get(
                    "name",
                    ""
                ).strip()

                or candidate.get(
                    "name",
                    ""
                ).strip()

                or "Unknown Candidate"
            )

            # --------------------------------------------------
            # DATA
            # --------------------------------------------------

            candidate_skills = resume_json.get(
                "skills",
                []
            )

            candidate_education = resume_json.get(
                "education",
                []
            )

            candidate_experience = resume_json.get(
                "experience",
                ""
            )

            # --------------------------------------------------
            # ENSURE LISTS
            # --------------------------------------------------

            if not isinstance(
                candidate_skills,
                list
            ):

                candidate_skills = []

            if not isinstance(
                candidate_education,
                list
            ):

                candidate_education = []

            # ==================================================
            # JD SKILLS
            # ==================================================

            jd_skills = [

                skill.strip()

                for skill in skills.split(",")

                if skill.strip()
            ]

            # ==================================================
            # ATS
            # ==================================================

            skill_score = scorer.skill_match(
                candidate_skills,
                jd_skills
            )

            experience_score = scorer.experience_match(
                candidate_experience,
                experience
            )

            education_text = " ".join(

                str(item)

                for item in candidate_education
            )

            education_score = scorer.education_match(
                education_text,
                education
            )

            # --------------------------------------------------
            # SEMANTIC SCORE
            # --------------------------------------------------

            similarity_score = float(

                candidate.get(
                    "score",
                    0
                )
            )

            ats_score = scorer.calculate(

                similarity=similarity_score,

                skill_score=skill_score,

                experience_score=experience_score,

                education_score=education_score
            )

            # ==================================================
            # STORE RESULT
            # ==================================================

            results.append({

                "Candidate":
                    candidate_name,

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
                    candidate_skills,

                "Education":
                    candidate_education,

                "Experience":
                    candidate_experience,

                "Projects":
                    resume_json.get(
                        "projects",
                        []
                    ),

                "Certifications":
                    resume_json.get(
                        "certifications",
                        []
                    ),

                "Resume Text":
                    candidate.get(
                        "text",
                        ""
                    ),

                "Semantic Similarity":
                    round(
                        float(
                            candidate.get(
                                "similarity",
                                0
                            )
                        ),
                        3
                    ),

                "Rerank Score":
                    round(
                        float(
                            candidate.get(
                                "rerank_score",
                                0
                            )
                        ),
                        3
                    ),

                "Skill Match":
                    round(
                        skill_score * 100,
                        2
                    ),

                "Experience Match":
                    round(
                        experience_score * 100,
                        2
                    ),

                "Education Match":
                    round(
                        education_score * 100,
                        2
                    ),

                "ATS Score":
                    ats_score
            })

            current_progress = (

                60

                +

                int(
                    20
                    *
                    (
                        (position + 1)
                        /
                        len(ranked)
                    )
                )
            )

            progress.progress(
                min(
                    current_progress,
                    80
                )
            )

        # ==================================================
        # DATAFRAME
        # ==================================================

        df = pd.DataFrame(
            results
        )

        df = df.sort_values(
            by="ATS Score",
            ascending=False
        ).reset_index(
            drop=True
        )

        # ==================================================
        # STAGE 5
        # LLM EVALUATION
        # ==================================================

        status.info(
            "🤖 Performing LLM evaluation..."
        )

        evaluation_limit = min(
            settings.FINAL_TOP_K,
            len(df)
        )

        recommendations = []

        for index in range(
            evaluation_limit
        ):

            resume_text = df.loc[
                index,
                "Resume Text"
            ]

            try:

                evaluation = evaluate_candidate(
                    resume_text,
                    jd_text
                )

                recommendation = evaluation.get(
                    "hiring_recommendation",
                    ""
                )

                suitable_role = evaluation.get(
                    "suitable_role",
                    ""
                )

                strengths = evaluation.get(
                    "strengths",
                    []
                )

                weaknesses = evaluation.get(
                    "weaknesses",
                    []
                )

                missing_skills = evaluation.get(
                    "missing_skills",
                    []
                )

            except Exception as e:

                recommendation = (
                    f"LLM evaluation failed: {e}"
                )

                suitable_role = ""

                strengths = []

                weaknesses = []

                missing_skills = []

            recommendations.append({

                "index":
                    index,

                "Recommendation":
                    recommendation,

                "Suitable Role":
                    suitable_role,

                "Strengths":
                    strengths,

                "Weaknesses":
                    weaknesses,

                "Missing Skills":
                    missing_skills
            })

        # ==================================================
        # ADD EVALUATION
        # ==================================================

        for item in recommendations:

            index = item["index"]

            df.loc[
                index,
                "Recommendation"
            ] = item["Recommendation"]

            df.loc[
                index,
                "Suitable Role"
            ] = item["Suitable Role"]

            df.loc[
                index,
                "Strengths"
            ] = ", ".join(
                str(x)
                for x in item["Strengths"]
            )

            df.loc[
                index,
                "Weaknesses"
            ] = ", ".join(
                str(x)
                for x in item["Weaknesses"]
            )

            df.loc[
                index,
                "Missing Skills"
            ] = ", ".join(
                str(x)
                for x in item["Missing Skills"]
            )

        # --------------------------------------------------
        # NOT EVALUATED
        # --------------------------------------------------

        for index in range(
            evaluation_limit,
            len(df)
        ):

            df.loc[
                index,
                "Recommendation"
            ] = "Not evaluated by LLM"

        # ==================================================
        # SAVE
        # ==================================================

        st.session_state[
            "screening_results"
        ] = df

        progress.progress(100)

        status.success(
            "🎉 Candidate screening completed successfully!"
        )

    # ======================================================
    # DISPLAY RESULTS
    # ======================================================

    if st.session_state[
        "screening_results"
    ] is not None:

        df = st.session_state[
            "screening_results"
        ]

        # ==================================================
        # RANKED CANDIDATES
        # ==================================================

        st.subheader(
            "🏆 Ranked Candidates"
        )

        display_columns = [

            "Candidate",

            "ATS Score",

            "Semantic Similarity",

            "Rerank Score",

            "Skill Match",

            "Experience Match",

            "Education Match",

            "Recommendation"
        ]

        available_columns = [

            column

            for column in display_columns

            if column in df.columns
        ]

        st.dataframe(
            df[available_columns],
            use_container_width=True,
            hide_index=True
        )

        # ==================================================
        # METRICS
        # ==================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Resumes",
                len(df)
            )

        with col2:

            st.metric(
                "Top Candidate",

                df.iloc[0]["Candidate"]

                if len(df) > 0

                else "-"
            )

        with col3:

            st.metric(
                "Average ATS",

                round(
                    df["ATS Score"].mean(),
                    2
                )

                if len(df) > 0

                else 0
            )

        with col4:

            st.metric(
                "Highest ATS",

                round(
                    df["ATS Score"].max(),
                    2
                )

                if len(df) > 0

                else 0
            )

        # ==================================================
        # ANALYTICS
        # ==================================================

        st.subheader(
            "📊 Candidate Analytics"
        )

        chart1 = ats_distribution(
            df
        )

        if chart1 is not None:

            st.plotly_chart(
                chart1,
                use_container_width=True
            )

        chart2 = top_skills(
            df
        )

        if chart2 is not None:

            st.plotly_chart(
                chart2,
                use_container_width=True
            )

        chart3 = experience_chart(
            df
        )

        if chart3 is not None:

            st.plotly_chart(
                chart3,
                use_container_width=True
            )

        # ==================================================
        # CANDIDATE DETAILS
        # ==================================================

        st.subheader(
            "👤 Candidate Details"
        )

        for index, row in df.iterrows():

            with st.expander(

                f"#{index + 1} "
                f"{row['Candidate']} "
                f"— ATS {row['ATS Score']}"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "**Email:**",
                        row.get(
                            "Email",
                            ""
                        )
                    )

                    st.write(
                        "**Phone:**",
                        row.get(
                            "Phone",
                            ""
                        )
                    )

                    st.write(
                        "**Location:**",
                        row.get(
                            "Location",
                            ""
                        )
                    )

                    st.write(
                        "**Experience:**",
                        row.get(
                            "Experience",
                            ""
                        )
                    )

                with col2:

                    st.write(
                        "**ATS Score:**",
                        row.get(
                            "ATS Score",
                            0
                        )
                    )

                    st.write(
                        "**Semantic Similarity:**",
                        row.get(
                            "Semantic Similarity",
                            0
                        )
                    )

                    st.write(
                        "**Rerank Score:**",
                        row.get(
                            "Rerank Score",
                            0
                        )
                    )

                    st.write(
                        "**Skill Match:**",
                        f"{row.get('Skill Match', 0)}%"
                    )

                # --------------------------------------------------
                # SKILLS
                # --------------------------------------------------

                skills_value = row.get(
                    "Skills",
                    []
                )

                if isinstance(
                    skills_value,
                    list
                ):

                    skills_text = ", ".join(
                        str(x)
                        for x in skills_value
                    )

                else:

                    skills_text = str(
                        skills_value
                    )

                st.write(
                    "**Skills:**",
                    skills_text
                )

                # --------------------------------------------------
                # EDUCATION
                # --------------------------------------------------

                education_value = row.get(
                    "Education",
                    []
                )

                if isinstance(
                    education_value,
                    list
                ):

                    education_text = ", ".join(
                        str(x)
                        for x in education_value
                    )

                else:

                    education_text = str(
                        education_value
                    )

                st.write(
                    "**Education:**",
                    education_text
                )

                # --------------------------------------------------
                # SUITABLE ROLE
                # --------------------------------------------------

                st.write(
                    "**Suitable Role:**",
                    row.get(
                        "Suitable Role",
                        ""
                    )
                )

                # --------------------------------------------------
                # STRENGTHS
                # --------------------------------------------------

                st.write(
                    "**Strengths:**",
                    row.get(
                        "Strengths",
                        ""
                    )
                )

                # --------------------------------------------------
                # WEAKNESSES
                # --------------------------------------------------

                st.write(
                    "**Weaknesses:**",
                    row.get(
                        "Weaknesses",
                        ""
                    )
                )

                # --------------------------------------------------
                # MISSING SKILLS
                # --------------------------------------------------

                st.write(
                    "**Missing Skills:**",
                    row.get(
                        "Missing Skills",
                        ""
                    )
                )

                # --------------------------------------------------
                # RECOMMENDATION
                # --------------------------------------------------

                st.write(
                    "**Hiring Recommendation:**",
                    row.get(
                        "Recommendation",
                        ""
                    )
                )

        # ==================================================
        # EXPORT
        # ==================================================

        st.subheader(
            "📥 Export Results"
        )

        try:

            export_df = df.drop(
                columns=["Resume Text"],
                errors="ignore"
            )

            excel_path = export_excel(
                export_df
            )

            with open(
                excel_path,
                "rb"
            ) as file:

                st.download_button(
                    "⬇ Download Excel Report",
                    data=file,
                    file_name="candidate_report.xlsx",
                    mime=(
                        "application/vnd."
                        "openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    )
                )

        except Exception as e:

            st.error(
                f"Excel export failed: {e}"
            )


# ======================================================
# TAB 3
# EVALUATION
# ======================================================

with tab3:

    st.header(
        "📊 Recruitment System Evaluation"
    )

    st.write(
        """
        Evaluate the quality of the candidate ranking
        using human-labelled ground truth.
        """
    )

    # ==================================================
    # CHECK SCREENING RESULTS
    # ==================================================

    if st.session_state[
        "screening_results"
    ] is None:

        st.info(
            "First analyze candidates in the "
            "'Resume Screening' tab."
        )

        st.stop()

    df = st.session_state[
        "screening_results"
    ]

    # ==================================================
    # GROUND TRUTH INFORMATION
    # ==================================================

    st.subheader(
        "1️⃣ Ground Truth"
    )

    st.write(
        """
        Upload a CSV containing human/recruiter
        evaluation of the candidates.
        """
    )

    st.code(
        """
Candidate,Relevance,Human ATS Score
John,5,90
Priya,4,80
Rahul,1,45
Anjali,5,88
        """,
        language="csv"
    )

    ground_truth_file = st.file_uploader(
        "Upload Ground Truth CSV",
        type=["csv"],
        key="ground_truth"
    )

    if ground_truth_file is None:

        st.warning(
            "Upload ground_truth.csv to calculate "
            "the evaluation metrics."
        )

        st.stop()

    # ==================================================
    # READ GROUND TRUTH
    # ==================================================

    try:

        ground_truth = pd.read_csv(
            ground_truth_file
        )

    except Exception as e:

        st.error(
            f"Could not read ground truth file: {e}"
        )

        st.stop()

    # ==================================================
    # VALIDATE COLUMNS
    # ==================================================

    required_columns = [
        "Candidate",
        "Relevance"
    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in ground_truth.columns
    ]

    if missing_columns:

        st.error(
            "Missing columns: "
            +
            ", ".join(missing_columns)
        )

        st.stop()

    # ==================================================
    # NORMALIZE NAMES
    # ==================================================

    ground_truth["Candidate"] = (
        ground_truth["Candidate"]
        .astype(str)
        .str.strip()
    )

    df["Candidate"] = (
        df["Candidate"]
        .astype(str)
        .str.strip()
    )

    # ==================================================
    # MERGE
    # ==================================================

    evaluation_df = df.merge(
        ground_truth,
        on="Candidate",
        how="inner"
    )

    if evaluation_df.empty:

        st.error(
            "No candidate names matched between "
            "screening results and ground truth."
        )

        st.stop()

    # ==================================================
    # SHOW MATCHED DATA
    # ==================================================

    st.subheader(
        "Matched Evaluation Data"
    )

    show_columns = [
        "Candidate",
        "ATS Score",
        "Relevance"
    ]

    if "Human ATS Score" in evaluation_df.columns:

        show_columns.append(
            "Human ATS Score"
        )

    st.dataframe(
        evaluation_df[
            show_columns
        ],
        use_container_width=True,
        hide_index=True
    )

    # ==================================================
    # TOP K
    # ==================================================

    st.subheader(
        "2️⃣ Ranking Metrics"
    )

    k = st.slider(
        "Select K",
        min_value=1,
        max_value=min(
            10,
            len(df)
        ),
        value=min(
            5,
            len(df)
        )
    )

    # ==================================================
    # SYSTEM RANKING
    # ==================================================

    ranked_candidates = (
        df["Candidate"]
        .tolist()
    )

    # ==================================================
    # RELEVANT CANDIDATES
    # ==================================================

    relevant_candidates = (
        ground_truth[
            ground_truth["Relevance"] >= 3
        ]["Candidate"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    # ==================================================
    # PRECISION@K
    # ==================================================

    precision = precision_at_k(
        ranked_candidates,
        relevant_candidates,
        k
    )

    # ==================================================
    # RECALL@K
    # ==================================================

    recall = recall_at_k(
        ranked_candidates,
        relevant_candidates,
        k
    )

    # ==================================================
    # MRR@K
    # ==================================================

    mrr = mrr_at_k(
        ranked_candidates,
        relevant_candidates,
        k
    )

    # ==================================================
    # NDCG@K
    # ==================================================

    relevance_dict = dict(
        zip(
            ground_truth["Candidate"],
            ground_truth["Relevance"]
        )
    )

    ndcg = candidate_ndcg_at_k(
        ranked_candidates,
        relevance_dict,
        k
    )

    # ==================================================
    # DISPLAY
    # ==================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            f"Precision@{k}",
            f"{precision:.3f}"
        )

    with col2:

        st.metric(
            f"Recall@{k}",
            f"{recall:.3f}"
        )

    with col3:

        st.metric(
            f"MRR@{k}",
            f"{mrr:.3f}"
        )

    with col4:

        st.metric(
            f"NDCG@{k}",
            f"{ndcg:.3f}"
        )

    # ==================================================
    # INTERPRETATION
    # ==================================================

    st.subheader(
        "📌 Metric Interpretation"
    )

    st.write(
        f"""
        **Precision@{k}:** {precision:.3f}

        Out of the top {k} candidates selected by
        the system, this represents the proportion
        that are relevant.

        **Recall@{k}:** {recall:.3f}

        This represents the proportion of all relevant
        candidates that were successfully retrieved
        in the top {k}.

        **MRR@{k}:** {mrr:.3f}

        This measures how high the first relevant
        candidate appears in the ranking.

        **NDCG@{k}:** {ndcg:.3f}

        This evaluates whether highly relevant
        candidates are placed near the top of the
        ranking.
        """
    )

    # ==================================================
    # ATS / HUMAN SCORE EVALUATION
    # ==================================================

    if "Human ATS Score" in evaluation_df.columns:

        st.subheader(
            "3️⃣ ATS Score Evaluation"
        )

        evaluation_df["Human ATS Score"] = (
            pd.to_numeric(
                evaluation_df[
                    "Human ATS Score"
                ],
                errors="coerce"
            )
        )

        evaluation_df["ATS Score"] = (
            pd.to_numeric(
                evaluation_df[
                    "ATS Score"
                ],
                errors="coerce"
            )
        )

        score_data = evaluation_df.dropna(
            subset=[
                "Human ATS Score",
                "ATS Score"
            ]
        )

        if len(score_data) >= 2:

            human_scores = (
                score_data[
                    "Human ATS Score"
                ].tolist()
            )

            system_scores = (
                score_data[
                    "ATS Score"
                ].tolist()
            )

            correlation = spearman_correlation(
                human_scores,
                system_scores
            )

            mae = score_mae(
                human_scores,
                system_scores
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Spearman Correlation",
                    f"{correlation:.3f}"
                )

            with col2:

                st.metric(
                    "MAE",
                    f"{mae:.3f}"
                )

            st.write(
                f"""
                **Spearman Correlation = {correlation:.3f}**

                This measures how closely your system's
                candidate ranking agrees with the human
                recruiter ranking.

                **MAE = {mae:.3f}**

                This represents the average absolute
                difference between the system ATS score
                and the human ATS score.
                """
            )

        else:

            st.warning(
                "At least two candidates with valid "
                "Human ATS Score values are required."
            )

    # ==================================================
    # EVALUATION SUMMARY
    # ==================================================

    st.subheader(
        "📋 Evaluation Summary"
    )

    summary = {

        "Metric": [

            f"Precision@{k}",

            f"Recall@{k}",

            f"MRR@{k}",

            f"NDCG@{k}"
        ],

        "Score": [

            round(
                precision,
                4
            ),

            round(
                recall,
                4
            ),

            round(
                mrr,
                4
            ),

            round(
                ndcg,
                4
            )
        ]
    }

    summary_df = pd.DataFrame(
        summary
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )