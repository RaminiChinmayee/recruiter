import json

from src.llm import ask_llm


# ======================================================
# EVALUATE CANDIDATE
# ======================================================

def evaluate_candidate(
    resume_text,
    jd_text
):

    prompt = f"""
You are an expert technical recruiter.

Evaluate the candidate resume against the job description.

Return ONLY valid JSON.

JSON structure:

{{
    "suitable_role": "",
    "strengths": [],
    "weaknesses": [],
    "missing_skills": [],
    "hiring_recommendation": ""
}}

Rules:

1. suitable_role:
   Give the most appropriate role for this candidate based ONLY
   on the resume and job description.

2. strengths:
   List the candidate's strongest relevant skills, experience,
   education or projects.

3. weaknesses:
   List relevant weaknesses or gaps.

4. missing_skills:
   List skills required by the JD that are not demonstrated
   in the resume.

5. hiring_recommendation:
   Give one of:
   - Strong Hire
   - Hire
   - Consider
   - Reject

IMPORTANT:

- Compare the resume directly against the JD.
- Do not hallucinate.
- Do not assume skills that are not present.
- Return JSON only.
- Do not return markdown.
- Do not use ```json.

JOB DESCRIPTION:

{jd_text}

CANDIDATE RESUME:

{resume_text}
"""


    response = ask_llm(
        prompt,
        temperature=0,
        json_mode=True
    )


    data = json.loads(
        response
    )


    # ==================================================
    # DEFAULT STRUCTURE
    # ==================================================

    result = {

        "suitable_role": "",

        "strengths": [],

        "weaknesses": [],

        "missing_skills": [],

        "hiring_recommendation": ""

    }


    for key in result:

        if key in data:

            result[key] = data[key]


    # ==================================================
    # VALIDATE LIST FIELDS
    # ==================================================

    for field in [
        "strengths",
        "weaknesses",
        "missing_skills"
    ]:

        if not isinstance(
            result[field],
            list
        ):

            result[field] = [
                str(result[field])
            ]


    return result