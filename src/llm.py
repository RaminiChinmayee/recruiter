from groq import Groq
from src.config import settings


if not settings.GROQ_API_KEY:

    raise ValueError(
        "Missing GROQ_API_KEY"
    )


client = Groq(
    api_key=settings.GROQ_API_KEY
)



# -------------------------------------------------------
# Generic LLM Function
# -------------------------------------------------------

def ask_llm(
        prompt,
        temperature=0.2
):

    response = client.chat.completions.create(

        model=settings.LLM_MODEL,

        messages=[

            {
                "role": "system",
                "content":
                """
You are an expert technical recruiter.
Generate ATS-friendly professional content.
Never hallucinate information.
"""
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=temperature

    )


    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )



# -------------------------------------------------------
# Generate Job Description
# -------------------------------------------------------

def generate_job_description(details):


    prompt = f"""

Create a professional ATS optimized Job Description.


Company Name:

{details.get("company_name")}



Job Title:

{details.get("role")}



Location:

{details.get("location")}



Experience Required:

{details.get("experience")}



Employment Type:

{details.get("employment_type")}



Work Mode:

{details.get("work_mode")}



Salary:

{details.get("salary")}



Education:

{details.get("education")}



Technical Skills:

{details.get("technical_skills")}



Soft Skills:

{details.get("soft_skills")}



Responsibilities:

{details.get("responsibilities")}



Benefits:

{details.get("benefits")}



Generate the following sections:


1. Company Overview

2. About the Role

3. Job Summary

4. Key Responsibilities

5. Required Technical Skills

6. Required Soft Skills

7. Educational Qualifications

8. Preferred Qualifications

9. Benefits

10. Salary Details

11. Work Mode

12. Employment Type

13. Application Process



Rules:

- Write like a professional recruiter
- Optimize for ATS systems
- Use bullet points
- Avoid unnecessary repetition
- Do not mention that AI generated it
- Return only the Job Description


"""


    return ask_llm(
        prompt,
        temperature=0.4
    )