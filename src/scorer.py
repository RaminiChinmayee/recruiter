import re


class ATSScorer:

    # --------------------------------------------------
    # Overall ATS Score
    # --------------------------------------------------

    def calculate(

        self,

        similarity,
        skill_score,
        experience_score,
        education_score

    ):

        ats = (

            similarity * 0.40

            +

            skill_score * 0.30

            +

            experience_score * 0.20

            +

            education_score * 0.10

        )


        return round(
            ats * 100,
            2
        )


    # --------------------------------------------------
    # Skill Match
    # --------------------------------------------------

    def skill_match(
        self,
        resume_skills,
        jd_skills
    ):

        resume = {

            str(x).strip().lower()

            for x in resume_skills

            if str(x).strip()

        }


        jd = {

            str(x).strip().lower()

            for x in jd_skills

            if str(x).strip()

        }


        if not jd:

            return 0.0


        matched = (
            resume.intersection(jd)
        )


        return (
            len(matched)
            /
            len(jd)
        )


    # --------------------------------------------------
    # Experience Match
    # --------------------------------------------------

    def experience_match(
        self,
        resume_exp,
        required_exp
    ):

        resume_exp = str(
            resume_exp or ""
        )

        required_exp = str(
            required_exp or ""
        )


        resume_numbers = re.findall(
            r"\d+(?:\.\d+)?",
            resume_exp
        )


        required_numbers = re.findall(
            r"\d+(?:\.\d+)?",
            required_exp
        )


        # No experience information
        if not resume_numbers:

            return 0.5


        # No required experience
        if not required_numbers:

            return 0.5


        candidate = float(
            resume_numbers[0]
        )


        required = float(
            required_numbers[0]
        )


        if required <= 0:

            return 1.0


        if candidate >= required:

            return 1.0


        return min(
            candidate / required,
            1.0
        )


    # --------------------------------------------------
    # Education Match
    # --------------------------------------------------

    def education_match(
        self,
        education,
        required
    ):

        education = str(
            education or ""
        ).lower()


        required = str(
            required or ""
        ).lower()


        if not education or not required:

            return 0.5


        education_keywords = {

            "b.tech": [
                "b.tech",
                "btech",
                "bachelor",
                "engineering"
            ],

            "m.tech": [
                "m.tech",
                "mtech",
                "master"
            ],

            "computer science": [
                "computer science",
                "cse"
            ],

            "engineering": [
                "engineering",
                "engineer"
            ]

        }


        for keywords in (
            education_keywords.values()
        ):

            education_match_found = any(

                keyword in education

                for keyword in keywords

            )


            required_match_found = any(

                keyword in required

                for keyword in keywords

            )


            if (
                education_match_found
                and
                required_match_found
            ):

                return 1.0


        return 0.5