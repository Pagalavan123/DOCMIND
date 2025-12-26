🧠 DOCMIND AI

Hallucination-Safe Financial Document Intelligence System
Powered by Gemini + RAG + Structured Extraction + Memory

DOCMIND AI is an enterprise-grade document assistant that analyzes payslips using:

Deterministic data extraction

Retrieval-Augmented Generation (RAG)

Knowledge memory for multi-turn reasoning

Strict hallucination control

This project is built for learning & experimentation.

📁 Project Structure
DOCMIND-ai/
│
├── main.py
├── Payslips.pdf        ← combine all your sample payslips into this file
├── requirements.txt
└── README.md

🛠️ Setup Instructions
1️⃣ Create Virtual Environment
python -m venv venv

2️⃣ Activate Environment

Windows (PowerShell):

venv\Scripts\activate


Mac / Linux:

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Gemini API Key (Learning Only)

PowerShell:

$env:GOOGLE_API_KEY="#@##"


Linux / Mac:

export GOOGLE_API_KEY="#@##"


⚠️ For production, use secret managers or environment vaults.

5️⃣ Prepare Your PDF

Combine all sample payslips into one PDF

Rename it to:

Payslips.pdf


Place it in the root directory

6️⃣ Run the System
python main.py

🧪 Example Queries
give me all month salary
add all month salary
why october net salary is low compare to september

🧱 Architecture Overview
Layer	Description
LLM	Gemini 2.5 Flash-Lite
Vector Store	FAISS (learning), Postgres + pgvector (production)
Memory	In-process memory (learning), PostgreSQL (production)
Auth	Demo mode (learning), User-level auth via Handloop
Safety	Structured extraction + validation
Reasoning	Multi-turn with knowledge memory

🚀 Production Recommendations
Component	Production Upgrade
LLM	Gemini Pro / Gemini 1.5
Vector DB	PostgreSQL + pgvector
Conversation History	PostgreSQL
User Auth	Handloop
Secrets	Vault / Cloud Secret Manager
Deployment	Docker + Cloud Run / EC2

🎓 Learning Disclaimer
This repository is for learning & experimentation only.
Not intended for direct production use without security hardening.
