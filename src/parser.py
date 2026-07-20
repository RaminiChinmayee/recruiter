import os
import pdfplumber
from docx import Document


def extract_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def extract_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_resumes(folder_path):
    resumes = {}

    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)

            if file.lower().endswith(".pdf"):
                resumes[file] = extract_pdf(full_path)

            elif file.lower().endswith(".docx"):
                resumes[file] = extract_docx(full_path)

            elif file.lower().endswith(".txt"):
                resumes[file] = extract_txt(full_path)

    return resumes