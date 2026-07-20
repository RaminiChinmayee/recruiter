import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API Key
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")

# Initialize Groq client
client = Groq(api_key=API_KEY)

# Groq Model
MODEL = "llama-3.3-70b-versatile"


# --------------------------------------------------
# Generate Job Description
# --------------------------------------------------

def generate_job_description(details):
    """
    details is a dictionary containing:
    role
    location
    experience
    employment_type
    work_mode
    salary
    notice_period
    education
    technical_skills
    soft_skills
    responsibilities
    benefits
    """

    prompt = f"""
You are an experienced HR Recruiter.

Generate a professional Job Description using the information below.

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

Salary Range:
{details["salary"]}

Notice Period:
{details["notice_period"]}

Education:
{details["education"]}

Technical Skills:
{details["technical_skills"]}

Soft Skills:
{details["soft_skills"]}

Roles & Responsibilities:
{details["responsibilities"]}

Benefits:
{details["benefits"]}

Generate the following sections.

1. About the Company
(Create a generic company introduction.)

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

• Job Summary must be at least 150 words.

• Use bullet points.

• Don't repeat skills.

• Use professional recruiter language.

• Return only the Job Description.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1200
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Error generating Job Description:\n{e}"


# --------------------------------------------------
# Generate Interview Questions
# --------------------------------------------------

def generate_questions(name, skills, projects):
    """
    Generates interview questions from candidate skills and projects.
    """

    skills_text = ", ".join(skills) if skills else "Not Mentioned"

    projects_text = ", ".join(projects) if projects else "No Projects"

    prompt = f"""
Candidate Name:
{name}

Technical Skills:
{skills_text}

Projects:
{projects_text}

Generate

• 5 Technical Interview Questions

• 5 Behavioural Questions
"""

    if projects:
        prompt += """

• 3 Project-based Questions
"""

    else:
        prompt += """

• Ask one question why the candidate has not included projects.
"""

    prompt += """

Rules

Only output questions.

No answers.

No explanations.

No numbering.

Each question should be on a new line.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=700
        )

        text = response.choices[0].message.content

        questions = [
            line.strip("-•1234567890. ").strip()
            for line in text.split("\n")
            if line.strip()
        ]

        return questions

    except Exception as e:

        return [f"Error generating questions: {e}"]


# --------------------------------------------------
# Testing
# --------------------------------------------------

if __name__ == "__main__":

    details = {
        "role": "Machine Learning Engineer",
        "location": "Hyderabad",
        "experience": "2-4 Years",
        "employment_type": "Full Time",
        "work_mode": "Hybrid",
        "salary": "8-12 LPA",
        "notice_period": "30 Days",
        "education": "B.Tech/M.Tech in CSE or related field",
        "technical_skills": "Python, SQL, Machine Learning, TensorFlow, PyTorch",
        "soft_skills": "Communication, Teamwork, Problem Solving",
        "responsibilities": "Develop ML models, Deploy APIs, Data preprocessing",
        "benefits": "Health Insurance, PF, Flexible Working Hours"
    }

    print("=" * 100)
    print("GENERATING JOB DESCRIPTION...")
    print("=" * 100)

    jd = generate_job_description(details)

    print(jd)

    print("\n")
    print("=" * 100)
    print("GENERATING INTERVIEW QUESTIONS...")
    print("=" * 100)

    questions = generate_questions(
        "John Doe",
        ["Python", "Machine Learning", "SQL", "TensorFlow"],
        ["Resume Screening System", "Chatbot using LLM"]
    )

    for q in questions:
        print("-", q)