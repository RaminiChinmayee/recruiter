import json
import re

from src.llm import ask_llm



# -------------------------------------------------------
# Extract JSON safely
# -------------------------------------------------------

def extract_json(response):

    if not response:
        return {}


    response = response.strip()


    # Remove markdown formatting

    response = response.replace(
        "```json",
        ""
    )

    response = response.replace(
        "```",
        ""
    )


    # Find JSON object

    start = response.find("{")

    end = response.rfind("}")


    if start == -1 or end == -1:

        return {}


    json_text = response[start:end+1]


    try:

        return json.loads(json_text)


    except json.JSONDecodeError:


        # Fix common LLM JSON issues

        json_text = re.sub(
            r",\s*}",
            "}",
            json_text
        )


        json_text = re.sub(
            r",\s*]",
            "]",
            json_text
        )


        try:

            return json.loads(json_text)

        except:

            return {}



# -------------------------------------------------------
# Resume Parser
# -------------------------------------------------------

def parse_resume(text):


    prompt = f"""

You are an expert resume parser.

Extract information from this resume.

Return ONLY valid JSON.

No markdown.
No explanation.


JSON format:

{{
"name":"",
"email":"",
"phone":"",
"location":"",
"skills":[],
"education":[],
"experience":"",
"projects":[],
"certifications":[]
}}


Resume:

{text}

"""


    response = ask_llm(
        prompt,
        temperature=0
    )


    # DEBUG
    print("\n====== GROQ RESPONSE ======")
    print(response)
    print("==========================\n")


    data = extract_json(response)



    # If LLM fails, return fallback

    if not data:


        return {

            "name":
            text.split("\n")[0]
            if text else "Unknown",


            "email":
            "",


            "phone":
            "",


            "location":
            "",


            "skills":
            [],


            "education":
            [],


            "experience":
            "",


            "projects":
            [],


            "certifications":
            []

        }



    return data