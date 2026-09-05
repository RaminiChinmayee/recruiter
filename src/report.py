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

from reportlab.lib import colors
from reportlab.lib.styles import (
    getSampleStyleSheet
)


OUTPUT_DIR = "outputs/reports"


def create_output_folder():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# --------------------------------------------------
# Excel
# --------------------------------------------------

def export_excel(
    dataframe,
    filename="candidate_results.xlsx"
):

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


# --------------------------------------------------
# Candidate PDF
# --------------------------------------------------

def create_candidate_pdf(
    candidate,
    filename=None
):

    create_output_folder()


    if filename is None:

        filename = (

            candidate.get(
                "Candidate",
                "candidate"
            )

            +

            "_report.pdf"

        )


    path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    document = SimpleDocTemplate(
        path
    )


    styles = (
        getSampleStyleSheet()
    )


    elements = []


    elements.append(

        Paragraph(
            "AI Resume Screening Report",
            styles["Title"]
        )

    )


    elements.append(
        Spacer(1, 20)
    )


    for key, value in candidate.items():

        if isinstance(
            value,
            list
        ):

            value = ", ".join(
                str(x)
                for x in value
            )


        elements.append(

            Paragraph(

                f"<b>{key}</b>: "
                f"{value}",

                styles["BodyText"]

            )

        )


        elements.append(
            Spacer(1, 10)
        )


    elements.append(

        Paragraph(

            "Generated on: "

            +

            datetime.now().strftime(
                "%d-%m-%Y"
            ),

            styles["Italic"]

        )

    )


    document.build(
        elements
    )


    return path


# --------------------------------------------------
# Recruitment Summary PDF
# --------------------------------------------------

def create_recruitment_report(
    dataframe,
    filename="recruitment_summary.pdf"
):

    create_output_folder()


    path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    document = SimpleDocTemplate(
        path
    )


    styles = (
        getSampleStyleSheet()
    )


    elements = []


    elements.append(

        Paragraph(
            "AI Recruitment Evaluation Report",
            styles["Title"]
        )

    )


    elements.append(
        Spacer(1, 20)
    )


    total = len(
        dataframe
    )


    elements.append(

        Paragraph(

            f"Total Candidates Screened: "
            f"{total}",

            styles["Heading2"]

        )

    )


    elements.append(
        Spacer(1, 15)
    )


    table_data = [

        [
            "Candidate",
            "ATS Score",
            "Similarity",
            "Recommendation"
        ]

    ]


    for _, row in dataframe.iterrows():

        table_data.append(

            [

                str(
                    row.get(
                        "Candidate",
                        "-"
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
                        "Semantic Similarity",
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


    table = Table(
        table_data,
        repeatRows=1
    )


    table.setStyle(

        TableStyle(

            [

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.black
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                )

            ]

        )

    )


    elements.append(
        table
    )


    elements.append(
        Spacer(1, 20)
    )


    elements.append(

        Paragraph(

            "This report was automatically "
            "generated using the AI-powered "
            "resume screening pipeline.",

            styles["BodyText"]

        )

    )


    document.build(
        elements
    )


    return path