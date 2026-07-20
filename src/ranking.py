import pandas as pd
import re


class ResumeRanker:

    @staticmethod
    def parse_resume(text):

        def extract(field):
            match = re.search(
                rf"{field}\s*:\s*(.*)",
                text,
                re.IGNORECASE
            )

            return match.group(1).strip() if match else ""

        return {

            "Name": extract("Name"),

            "Email": extract("Email"),

            "Phone": extract("Phone"),

            "Location": extract("Location"),

            "Experience": extract("Experience"),

            "Skills": extract("Skills"),

            "Education": extract("Education"),

            "Projects": extract("Projects"),

            "Work Experience": extract("Work Experience")
        }

    @staticmethod
    def shortlist(results):

        shortlisted = []

        for candidate in results:

            data = ResumeRanker.parse_resume(candidate["text"])

            data["Resume"] = candidate["name"]

            data["Similarity Score"] = round(
                candidate["score"],
                3
            )

            shortlisted.append(data)

        return pd.DataFrame(shortlisted)

    @staticmethod
    def save(df, filename="outputs/shortlisted_candidates.xlsx"):

        df.to_excel(filename, index=False)

        return filename