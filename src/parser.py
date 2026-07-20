import re
import os
import pandas as pd


class ResumeRanker:

    # -----------------------------------------------------
    # Generic field extractor
    # -----------------------------------------------------

    @staticmethod
    def extract_field(field, text):

        pattern = rf"{field}\s*:\s*(.*)"

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

        return ""

    # -----------------------------------------------------
    # Parse Resume
    # -----------------------------------------------------

    @staticmethod
    def parse_resume(text):

        return {

            "Name":
            ResumeRanker.extract_field(
                "Name",
                text
            ),

            "Email":
            ResumeRanker.extract_field(
                "Email",
                text
            ),

            "Phone":
            ResumeRanker.extract_field(
                "Phone",
                text
            ),

            "Location":
            ResumeRanker.extract_field(
                "Location",
                text
            ),

            "Experience":
            ResumeRanker.extract_field(
                "Experience",
                text
            ),

            "Skills":
            ResumeRanker.extract_field(
                "Skills",
                text
            ),

            "Education":
            ResumeRanker.extract_field(
                "Education",
                text
            ),

            "Projects":
            ResumeRanker.extract_field(
                "Projects",
                text
            ),

            "Work Experience":
            ResumeRanker.extract_field(
                "Work Experience",
                text
            )
        }

    # -----------------------------------------------------
    # Extract Technical Skills from JD
    # -----------------------------------------------------

    @staticmethod
    def extract_jd_skills(jd_text):

        keywords = [

            "python",
            "java",
            "c",
            "c++",
            "sql",
            "mysql",
            "postgresql",
            "mongodb",
            "machine learning",
            "deep learning",
            "tensorflow",
            "pytorch",
            "keras",
            "nlp",
            "llm",
            "langchain",
            "rag",
            "transformers",
            "huggingface",
            "docker",
            "kubernetes",
            "git",
            "linux",
            "aws",
            "azure",
            "gcp",
            "flask",
            "fastapi",
            "django",
            "streamlit",
            "pandas",
            "numpy",
            "opencv",
            "power bi",
            "excel"
        ]

        jd_lower = jd_text.lower()

        found = []

        for skill in keywords:

            if skill in jd_lower:
                found.append(skill)

        return sorted(list(set(found)))

    # -----------------------------------------------------
    # Compare Resume Skills
    # -----------------------------------------------------

    @staticmethod
    def compare_skills(
        resume_skills,
        jd_skills
    ):

        resume_set = {

            x.strip().lower()

            for x in resume_skills.split(",")

            if x.strip()
        }

        jd_set = {

            x.lower()

            for x in jd_skills
        }

        matched = sorted(
            resume_set.intersection(jd_set)
        )

        missing = sorted(
            jd_set.difference(resume_set)
        )

        if len(jd_set) == 0:

            percentage = 0

        else:

            percentage = round(

                len(matched)

                / len(jd_set)

                * 100,

                2
            )

        return matched, missing, percentage

    # -----------------------------------------------------
    # Create Shortlisted DataFrame
    # -----------------------------------------------------

    @staticmethod
    def shortlist(
        results,
        jd_text
    ):

        jd_skills = ResumeRanker.extract_jd_skills(
            jd_text
        )

        rows = []

        for candidate in results:

            data = ResumeRanker.parse_resume(
                candidate["text"]
            )

            matched, missing, percent = \
                ResumeRanker.compare_skills(

                    data.get("Skills", ""),

                    jd_skills
                )

            rows.append({

                "Resume":
                candidate["name"],

                "Candidate":
                data.get("Name", ""),

                "Email":
                data.get("Email", ""),

                "Phone":
                data.get("Phone", ""),

                "Experience":
                data.get("Experience", ""),

                "Skills":
                data.get("Skills", ""),

                "Matched Skills":
                ", ".join(matched),

                "Missing Skills":
                ", ".join(missing),

                "Skill Match %":
                percent,

                "Similarity Score":
                round(candidate["score"], 3)

            })

        df = pd.DataFrame(rows)

        if not df.empty:

            df = df.sort_values(

                "Similarity Score",

                ascending=False
            )

        return df

    # -----------------------------------------------------
    # Save Excel
    # -----------------------------------------------------

    @staticmethod
    def save(

        dataframe,

        filename="outputs/shortlisted_candidates.xlsx"

    ):

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        dataframe.to_excel(

            filename,

            index=False
        )

        return filename