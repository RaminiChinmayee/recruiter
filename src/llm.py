import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "openai/gpt-oss-120b"
# ======================================================
# GENERIC LLM CALL
# ======================================================

def ask_llm(
    prompt,
    temperature=0,
    json_mode=False
):

    kwargs = {
        "model": MODEL_NAME,

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": temperature
    }

    if json_mode:
        kwargs["response_format"] = {
            "type": "json_object"
        }

    response = client.chat.completions.create(
        **kwargs
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "LLM returned an empty response."
        )

    return content


# ======================================================
# JOB DESCRIPTION GENERATOR
# ======================================================

def generate_job_description(details):

    prompt = f"""
Create a professional ATS-optimized job description.

Company:
{details["company_name"]}

Role:
{details["role"]}

Experience:
{details["experience"]}

Location:
{details["location"]}

Technical Skills:
{details["technical_skills"]}

Education:
{details["education"]}

Responsibilities:
{details["responsibilities"]}

Soft Skills:
{details["soft_skills"]}

Work Mode:
{details["work_mode"]}

Employment Type:
{details["employment_type"]}

Salary:
{details["salary"]}

Benefits:
{details["benefits"]}

Create a clear professional job description with:

1. Job Title
2. Company
3. Location
4. Experience
5. Employment Type
6. Job Summary
7. Responsibilities
8. Required Technical Skills
9. Education
10. Soft Skills
11. Salary
12. Benefits

Do not invent company-specific information.
"""

    return ask_llm(
        prompt,
        temperature=0.2
    )