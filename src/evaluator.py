from src.llm import ask_llm



def evaluate_candidate(
    resume,
    jd
):


    prompt=f"""

You are a senior technical recruiter.

Compare candidate resume with job description.


JOB DESCRIPTION:

{jd}


RESUME:

{resume}



Return:

1. Candidate Strengths

2. Weaknesses

3. Missing Skills

4. Suitable Role

5. Hiring Recommendation


Use professional recruiter language.

"""


    return ask_llm(
        prompt,
        temperature=0.2
    )