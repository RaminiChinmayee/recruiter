# 🤖 AI Resume Screening System

An AI-powered resume screening and ranking system that automatically parses resumes, generates job descriptions, and shortlists the most relevant candidates using embeddings and semantic search.

---

## 🚀 Features

* 📄 Upload and parse multiple resumes (PDF/Text)
* 🧠 AI-generated Job Description based on requirements
* 🔍 Semantic resume search using embeddings
* 📊 Intelligent candidate ranking system
* ⚡ Fast similarity search using FAISS index
* 🖥️ Interactive web UI built with Streamlit

---

## 🏗️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **NLP Model:** Sentence Transformers / Embeddings
* **Vector Database:** FAISS
* **File Parsing:** pdfplumber / custom parsers
* **ML Techniques:** Cosine similarity, embedding-based retrieval

---

## 📂 Project Structure

```
recruiter/
│── app.py
│── src/
│   ├── parser.py          # Resume loading & parsing
│   ├── analyzer.py        # Text cleaning & preprocessing
│   ├── retriever.py       # FAISS-based search
│   ├── ranking.py         # Candidate ranking logic
│   ├── llm.py             # Job description generation
│── data/
│   └── resumes/           # Uploaded resumes
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/resume-screening-ai.git
cd resume-screening-ai
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🧠 How It Works

1. **Upload Resumes** → PDFs are parsed into raw text
2. **Preprocessing** → Text is cleaned and normalized
3. **Embedding Generation** → Sentence Transformers convert text into vectors
4. **FAISS Indexing** → Efficient similarity search is built
5. **Job Description Input** → User provides or AI generates JD
6. **Ranking** → Candidates ranked based on semantic similarity

---

## 📊 Output

* Ranked list of candidates
* Similarity scores
* Top matching resumes for a job description

---

## 📌 Future Improvements

* 🔥 Add LLM-based resume summarization
* 📈 Dashboard analytics for recruiters
* 🧾 Multi-role job support
* ☁️ Cloud deployment (AWS / Azure)
* 🧑‍💼 Candidate skill extraction

---

## 🐛 Common Issues

* **FAISS error:** Ensure correct installation (`pip install faiss-cpu`)
* **PDF parsing error:** Install `pdfplumber`
* **Embedding mismatch:** Check model consistency

---

