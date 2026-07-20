import re


def extract(field, text):

    m = re.search(f"{field}: (.*)", text)

    return m.group(1).strip() if m else ""


def parse_resume(text):

    return {

        "Name":extract("Name",text),

        "Email":extract("Email",text),

        "Phone":extract("Phone",text),

        "Location":extract("Location",text),

        "Experience":extract("Experience",text),

        "Skills":extract("Skills",text),

        "Projects":extract("Projects",text),

        "Education":extract("Education",text)

    }