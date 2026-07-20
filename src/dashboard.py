import pandas as pd
import plotly.express as px


def create_dataframe(results):

    return pd.DataFrame(results)



def ats_distribution(df):

    fig = px.histogram(
        df,
        x="ATS Score",
        nbins=10,
        title="ATS Score Distribution"
    )

    return fig



def top_skills(df):

    skills=[]


    for skill_list in df["Skills"]:

        if isinstance(skill_list,list):

            skills.extend(skill_list)


    skill_count=pd.Series(
        skills
    ).value_counts().reset_index()


    skill_count.columns=[
        "Skill",
        "Count"
    ]


    fig=px.bar(

        skill_count,

        x="Skill",

        y="Count",

        title="Top Candidate Skills"

    )


    return fig



def experience_chart(df):


    fig=px.histogram(

        df,

        x="Experience",

        title="Experience Distribution"

    )


    return fig