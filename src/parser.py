import json

from src.llm import ask_llm


def empty_resume():
    return {
        "name": "",
        "email": "",
        "phone": "",
        "location": "",
        "skills": [],
        "education": [],
        "experience": "",
        "projects": [],
        "certifications": []
    }


def parse_resume(text):
    """
    Extract structured information from a resume.
    """

    if not text or not text.strip():
        return empty_resume()

    prompt = f"""
Extract structured information from the following candidate resume.

Return ONLY a valid JSON object using exactly this structure:

{{
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
    "skills": [],
    "education": [],
    "experience": "",
    "projects": [],
    "certifications": []
}}

Rules:

- Extract only information explicitly present in the resume.
- Never hallucinate.
- Do not infer missing information.
- Use an empty string when a string field is unavailable.
- Use an empty list when a list field is unavailable.
- skills must be an array of strings.
- education must be an array of strings.
- projects must be an array of strings.
- certifications must be an array of strings.
- experience should be a concise description of the candidate's experience.
- Return valid JSON only.

RESUME:
{text}
"""

    try:
        print("Calling LLM for resume parsing...")

        response = ask_llm(
            prompt,
            temperature=0,
            json_mode=True
        )

        print("Resume parsing completed.")

        data = json.loads(response)

        default = empty_resume()

        # Make sure every expected key exists
        for key in default:
            if key not in data:
                data[key] = default[key]

        # Make sure list fields are actually lists
        list_fields = [
            "skills",
            "education",
            "projects",
            "certifications"
        ]

        for field in list_fields:
            if not isinstance(data[field], list):
                data[field] = []

        # Make sure string fields are strings
        string_fields = [
            "name",
            "email",
            "phone",
            "location",
            "experience"
        ]

        for field in string_fields:
            if not isinstance(data[field], str):
                data[field] = str(data[field])

        return data

    except json.JSONDecodeError as e:
        print(f"Resume parser returned invalid JSON: {e}")
        return empty_resume()

    except Exception as e:
        print(f"Resume parsing error: {e}")
        return empty_resume()