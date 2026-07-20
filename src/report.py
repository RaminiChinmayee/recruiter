# src/report.py

import os
from datetime import datetime

import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet



OUTPUT_DIR = "outputs/reports"



def create_output_folder():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )



# -------------------------------------------------------
# Excel Report
# -------------------------------------------------------

def export_excel(
        dataframe,
        filename="candidate_results.xlsx"
):

    """
    Export candidate ranking data to Excel.
    """

    create_output_folder()


    path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    dataframe.to_excel(
        path,
        index=False
    )


    return path



# -------------------------------------------------------
# Candidate PDF Report
# -------------------------------------------------------

def create_candidate_pdf(
        candidate,
        filename=None
):

    """
    Generate single candidate evaluation PDF.

    candidate example:

    {
        "Name":"John",
        "ATS Score":90,
        "Skills":["Python","SQL"],
        "Strengths":"...",
        "Weaknesses":"..."
    }

    """

    create_output_folder()


    if filename is None:

        filename = (
            candidate.get(
                "Name",
                "candidate"
            )
            +
            "_report.pdf"
        )


    path=os.path.join(
        OUTPUT_DIR,
        filename
    )


    document=SimpleDocTemplate(
        path
    )


    styles=getSampleStyleSheet()


    elements=[]


    title=Paragraph(

        "AI Resume Screening Report",

        styles["Title"]

    )


    elements.append(title)

    elements.append(
        Spacer(1,20)
    )


    for key,value in candidate.items():


        if isinstance(value,list):

            value=", ".join(value)


        text=f"""

        <b>{key}</b> :

        {value}

        """


        elements.append(

            Paragraph(

                text,

                styles["BodyText"]

            )

        )


        elements.append(

            Spacer(
                1,
                12
            )

        )



    elements.append(

        Paragraph(

            f"""
            Generated on:
            {datetime.now().strftime("%d-%m-%Y")}
            """,

            styles["Italic"]

        )

    )


    document.build(
        elements
    )


    return path



# -------------------------------------------------------
# Complete Recruitment Report
# -------------------------------------------------------

def create_recruitment_report(
        dataframe,
        filename="recruitment_summary.pdf"
):

    """
    Creates recruiter dashboard PDF.

    Contains:

    - Total candidates
    - Top candidates
    - ATS scores
    - Ranking table

    """


    create_output_folder()



    path=os.path.join(
        OUTPUT_DIR,
        filename
    )


    document=SimpleDocTemplate(
        path
    )


    styles=getSampleStyleSheet()


    elements=[]



    elements.append(

        Paragraph(

            "AI Recruitment Evaluation Report",

            styles["Title"]

        )

    )


    elements.append(
        Spacer(1,20)
    )



    total=len(dataframe)



    elements.append(

        Paragraph(

            f"""
            Total Candidates Screened:
            {total}
            """,

            styles["Heading2"]

        )

    )


    elements.append(
        Spacer(1,15)
    )



    # Create table data


    table_data=[

        [

        "Candidate",

        "ATS Score",

        "Similarity",

        "Status"

        ]

    ]



    for _,row in dataframe.iterrows():


        table_data.append(

            [

            str(
                row.get(
                    "Candidate",
                    row.get(
                        "Resume",
                        ""
                    )
                )
            ),

            str(
                row.get(
                    "ATS Score",
                    "-"
                )
            ),

            str(
                row.get(
                    "Similarity Score",
                    "-"
                )
            ),

            str(
                row.get(
                    "Recommendation",
                    "-"
                )
            )

            ]

        )



    table=Table(
        table_data,
        repeatRows=1
    )


    table.setStyle(

        TableStyle(

            [

            ("GRID",
             (0,0),
             (-1,-1),
             0.5,
             None),


            ("VALIGN",
             (0,0),
             (-1,-1),
             "TOP")

            ]

        )

    )


    elements.append(
        table
    )


    elements.append(
        Spacer(1,20)
    )


    elements.append(

        Paragraph(

            """
            This report was automatically
            generated using GenAI based
            resume screening pipeline.
            """,

            styles["BodyText"]

        )

    )


    document.build(
        elements
    )


    return path