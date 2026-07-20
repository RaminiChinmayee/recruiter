import re



class ATSScorer:



    def calculate(

        self,

        similarity,

        skill_score,

        experience_score,

        education_score

    ):


        ats=(

            similarity*0.40

            +

            skill_score*0.30

            +

            experience_score*0.20

            +

            education_score*0.10

        )


        return round(

            ats*100,

            2

        )




    def skill_match(
        self,
        resume_skills,
        jd_skills
    ):


        resume=set(

            x.lower()

            for x in resume_skills

        )


        jd=set(

            x.lower()

            for x in jd_skills

        )


        if not jd:

            return 0



        return (

            len(
                resume.intersection(jd)
            )

            /

            len(jd)

        )




    def experience_match(
        self,
        resume_exp,
        required_exp
    ):


        numbers=re.findall(

            r'\d+',

            resume_exp

        )


        if not numbers:

            return 0.5



        candidate=int(
            numbers[0]
        )


        required=int(
            required_exp
        )


        if candidate>=required:

            return 1


        return candidate/required




    def education_match(
        self,
        education,
        required
    ):


        education=education.lower()

        required=required.lower()


        keywords=[

            "b.tech",
            "m.tech",
            "computer science",
            "engineering"

        ]


        for word in keywords:

            if word in education and word in required:

                return 1


        return 0.5