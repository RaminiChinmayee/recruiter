import pdfplumber
from docx import Document


# --------------------------------------------------
# PDF Extraction
# --------------------------------------------------

def extract_pdf(file):

    text_parts = []


    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text_parts.append(
                    page_text
                )


    return "\n".join(text_parts)


# --------------------------------------------------
# DOCX Extraction
# --------------------------------------------------

def extract_docx(file):

    document = Document(file)


    paragraphs = [

        para.text.strip()

        for para in document.paragraphs

        if para.text.strip()

    ]


    return "\n".join(paragraphs)


# --------------------------------------------------
# TXT Extraction
# --------------------------------------------------

def extract_txt(file):

    return file.read().decode(
        "utf-8",
        errors="ignore"
    )


# --------------------------------------------------
# Single File Extraction
# --------------------------------------------------

def extract_uploaded_file(uploaded_file):

    extension = (
        uploaded_file.name
        .split(".")[-1]
        .lower()
    )


    uploaded_file.seek(0)


    if extension == "pdf":

        return extract_pdf(
            uploaded_file
        )


    elif extension == "docx":

        return extract_docx(
            uploaded_file
        )


    elif extension == "txt":

        return extract_txt(
            uploaded_file
        )


    else:

        raise ValueError(
            f"Unsupported file format: {extension}"
        )


# --------------------------------------------------
# Multiple Resume Extraction
# --------------------------------------------------

def load_multiple_resumes(files):

    resumes = {}


    for file in files:

        try:

            text = extract_uploaded_file(
                file
            )


            if text and text.strip():

                resumes[file.name] = (
                    text.strip()
                )


        except Exception as e:

            print(
                f"Error reading "
                f"{file.name}: {e}"
            )


    return resumes