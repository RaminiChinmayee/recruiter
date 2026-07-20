import re


def clean_text(text: str) -> str:
    """
    Cleans extracted text.
    """

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def split_skills(skill_string):
    """
    Converts comma separated skills to list.
    """

    if not skill_string:
        return []

    return [

        skill.strip()

        for skill in skill_string.split(",")

        if skill.strip()

    ]


def normalize_skill(skill):
    """
    Lowercase skill.
    """

    return skill.lower().strip()


def similarity_to_percentage(score):
    """
    Converts cosine similarity
    to percentage.
    """

    return round(score * 100, 2)