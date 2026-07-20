import os
from groq import Groq
from dotenv import load_dotenv

# -------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=API_KEY)

MODEL = "llama-3.3-70b-versatile"


# -------------------------------------------------------
# Generic LLM Function
# -------------------------------------------------------

def ask_llm(prompt, temperature=0.7, max_tokens=1200):
    """
    Sends a prompt to Groq and returns the response.
    """

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an experienced HR Recruiter and Technical Interviewer. "
                        "Generate ATS-friendly, professional and structured responses."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()

    except Exception as e:

        return f"Error: {str(e)}"


# -------------------------------------------------------
# Generate Job Description
# -------------------------------------------------------

def generate_job_description(details):
    """
    details = {
        role,
        location,
        experience,
        employment_type,
        work_mode,
        salary,
        notice_period,
        education,
        technical_skills,
        soft_skills,
        responsibilities,
        benefits
    }
    """

    prompt = f"""
Generate a professional ATS-friendly Job Description.

Job Title:
{details["role"]}

Location:
{details["location"]}

Experience Required:
{details["experience"]}

Employment Type:
{details["employment_type"]}

Work Mode:
{details["work_mode"]}

Salary:
{details["salary"]}

Notice Period:
{details["notice_period"]}

Education:
{details["education"]}

Technical Skills:
{details["technical_skills"]}

Soft Skills:
{details["soft_skills"]}

Responsibilities:
{details["responsibilities"]}

Benefits:
{details["benefits"]}

Generate the following sections.

1. About the Company

2. Job Summary

3. Key Responsibilities

4. Required Technical Skills

5. Required Soft Skills

6. Educational Qualifications

7. Preferred Qualifications

8. Benefits

9. Salary

10. Notice Period

11. Employment Type

12. Work Mode

Rules

• Job Summary should be around 150 words.

• Use bullet points.

• Don't repeat skills.

• Use professional recruiter language.

• Return ONLY the Job Description.
"""

    return ask_llm(
        prompt,
        temperature=0.7,
        max_tokens=1400
    )


# -------------------------------------------------------
# Resume Summary
# -------------------------------------------------------

def summarize_resume(resume_text):
    """
    Summarizes a resume.
    """

    prompt = f"""
Summarize the following resume.

Resume

{resume_text}

Return

1. Candidate Summary

2. Technical Skills

3. Soft Skills

4. Experience

5. Education

6. Projects

7. Strengths

8. Weaknesses

9. Suitable Roles

Keep the summary under 250 words.
"""

    return ask_llm(
        prompt,
        temperature=0.4,
        max_tokens=700
    )


# -------------------------------------------------------
# Candidate Evaluation
# -------------------------------------------------------

def evaluate_candidate(resume_text, jd_text):
    """
    Evaluates resume against JD.
    """

    prompt = f"""
You are an experienced HR Recruiter.

Job Description

{jd_text}

Candidate Resume

{resume_text}

Evaluate the candidate.

Return

Overall Match Score (0-100)

Matched Skills

Missing Skills

Strengths

Weaknesses

Hiring Recommendation

Explain each section briefly.
"""

    return ask_llm(
        prompt,
        temperature=0.3,
        max_tokens=900
    )


# -------------------------------------------------------
# Generate Interview Questions
# -------------------------------------------------------

def generate_interview_questions(
    name,
    skills,
    projects
):
    """
    Generates interview questions.
    """

    skills_text = ", ".join(skills) if skills else "Not Mentioned"

    projects_text = ", ".join(projects) if projects else "No Projects"

    prompt = f"""
Candidate Name

{name}

Technical Skills

{skills_text}

Projects

{projects_text}

Generate

5 Technical Interview Questions

5 Behavioural Questions

"""

    if projects:

        prompt += """
3 Project Based Questions
"""

    else:

        prompt += """
Ask one question about why no projects are included.
"""

    prompt += """

Rules

Return ONLY questions.

No numbering.

No explanations.

One question per line.
"""

    text = ask_llm(
        prompt,
        temperature=0.5,
        max_tokens=700
    )

    questions = []

    for line in text.split("\n"):

        line = line.strip()

        if line:

            line = line.lstrip("-•1234567890. ")

            questions.append(line)

    return questions


# -------------------------------------------------------
# Test
# -------------------------------------------------------

if __name__ == "__main__":

    details = {
        "role": "Machine Learning Engineer",
        "location": "Hyderabad",
        "experience": "2-4 Years",
        "employment_type": "Full Time",
        "work_mode": "Hybrid",
        "salary": "8-12 LPA",
        "notice_period": "30 Days",
        "education": "B.Tech/M.Tech in Computer Science",
        "technical_skills": "Python, SQL, Machine Learning, TensorFlow, PyTorch",
        "soft_skills": "Communication, Teamwork, Problem Solving",
        "responsibilities": "Build ML models, Deploy APIs, Data preprocessing",
        "benefits": "Health Insurance, Flexible Hours, PF"
    }

    print("=" * 80)
    print("JOB DESCRIPTION")
    print("=" * 80)

    jd = generate_job_description(details)

    print(jd)

    print("\n")

    print("=" * 80)
    print("INTERVIEW QUESTIONS")
    print("=" * 80)

    questions = generate_interview_questions(
        "John Doe",
        ["Python", "SQL", "Machine Learning", "TensorFlow"],
        ["Resume Screening System", "Chatbot using LLM"]
    )

    for q in questions:
        print("-", q)