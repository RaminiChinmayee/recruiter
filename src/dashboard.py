import plotly.express as px
import pandas as pd


# ======================================================
# ATS DISTRIBUTION
# ======================================================

def ats_distribution(df):

    if df.empty:
        return None

    if "ATS Score" not in df.columns:
        return None

    data = df[
        ["Candidate", "ATS Score"]
    ].copy()

    data["ATS Score"] = pd.to_numeric(
        data["ATS Score"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["ATS Score"]
    )

    if data.empty:
        return None

    fig = px.bar(
        data,
        x="Candidate",
        y="ATS Score",
        title="ATS Score Distribution",
        text="ATS Score"
    )

    fig.update_layout(
        xaxis_title="Candidate",
        yaxis_title="ATS Score",
        xaxis_tickangle=-45
    )

    return fig


# ======================================================
# TOP SKILLS
# ======================================================

def top_skills(df):

    if df.empty:
        return None

    if "Skills" not in df.columns:
        return None

    all_skills = []

    for skills in df["Skills"]:

        if isinstance(skills, list):

            for skill in skills:

                if skill:

                    all_skills.append(
                        str(skill).strip()
                    )


    if not all_skills:
        return None


    skill_counts = (
        pd.Series(all_skills)
        .value_counts()
        .head(10)
        .reset_index()
    )


    skill_counts.columns = [
        "Skill",
        "Count"
    ]


    fig = px.bar(
        skill_counts,
        x="Skill",
        y="Count",
        title="Top Candidate Skills",
        text="Count"
    )

    fig.update_layout(
        xaxis_title="Skill",
        yaxis_title="Number of Candidates",
        xaxis_tickangle=-45
    )

    return fig


# ======================================================
# EXPERIENCE CHART
# ======================================================

def experience_chart(df):

    if df.empty:
        return None

    if "Experience Match" not in df.columns:
        return None

    data = df[
        ["Candidate", "Experience Match"]
    ].copy()

    data["Experience Match"] = pd.to_numeric(
        data["Experience Match"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Experience Match"]
    )

    if data.empty:
        return None


    fig = px.bar(
        data,
        x="Candidate",
        y="Experience Match",
        title="Experience Match",
        text="Experience Match"
    )

    fig.update_layout(
        xaxis_title="Candidate",
        yaxis_title="Experience Match (%)",
        xaxis_tickangle=-45
    )

    return fig