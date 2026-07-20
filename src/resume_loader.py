import pdfplumber
from docx import Document



# -------------------------------------------------------
# Extract PDF
# -------------------------------------------------------

def extract_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"


    return text



# -------------------------------------------------------
# Extract DOCX
# -------------------------------------------------------

def extract_docx(file):

    document = Document(file)


    text = "\n".join(

        para.text

        for para in document.paragraphs

        if para.text.strip()

    )


    return text



# -------------------------------------------------------
# Extract TXT
# -------------------------------------------------------

def extract_txt(file):

    return file.read().decode(
        "utf-8",
        errors="ignore"
    )



# -------------------------------------------------------
# Extract any uploaded file
# -------------------------------------------------------

def extract_uploaded_file(uploaded_file):

    extension = uploaded_file.name.split(".")[-1].lower()


    uploaded_file.seek(0)


    if extension == "pdf":

        return extract_pdf(uploaded_file)



    elif extension == "docx":

        return extract_docx(uploaded_file)



    elif extension == "txt":

        return extract_txt(uploaded_file)



    else:

        raise ValueError(
            "Unsupported file format"
        )



# -------------------------------------------------------
# Load multiple resumes
# -------------------------------------------------------

def load_multiple_resumes(files):

    resumes = {}


    for file in files:

        try:

            text = extract_uploaded_file(file)


            if text.strip():

                resumes[file.name] = text


        except Exception as e:

            print(
                f"Error reading {file.name}: {e}"
            )


    return resumes