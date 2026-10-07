# ⚖️ AI Legal Aid — Multi-Agent System

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5%2B-F5A623)
![IBM watsonx](https://img.shields.io/badge/IBM%20watsonx.ai-Granite-0530AD?logo=ibm&logoColor=white)

An AI-powered legal document analysis platform built with a multi-agent architecture.
Upload any legal agreement and receive instant risk analysis, statutory compliance checking,
a plain-English summary, and an interactive Q&A — all powered by IBM Granite on watsonx.ai.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Dashboard                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐  │
│  │ Document │ │  Multi-  │ │  Risk    │ │ Legal │  │
│  │  Upload  │ │  Agent   │ │Breakdown │ │  Q&A  │  │
│  │  (Tab 1) │ │ Analysis │ │  (Tab 3) │ │(Tab 4)│  │
│  └──────────┘ │  (Tab 2) │ └──────────┘ └───┬───┘  │
│               └─────┬────┘                  │       │
└─────────────────────┼──────────────────────-┼───────┘
                       │                       │
          ┌────────────┼────────────┐          │
          ▼            ▼            ▼          ▼
   ┌─────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
   │  Agent 1    │ │ Agent 2  │ │ Agent 4  │ │   Agent 3    │
   │  Contract   │ │Compliance│ │Summarizer│ │  RAG Q&A     │
   │  Reviewer   │ │ Checker  │ │          │ │              │
   └──────┬──────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘
          │              │             │              │
          └──────────────┴─────────────┴──────────────┘
                                │
                    ┌───────────┴──────────┐
                    │   utils/granite.py   │
                    │  GraniteClient       │
                    │  (+ MockLLM fallback)│
                    └───────────┬──────────┘
                                │
                    ┌───────────┴──────────┐
                    │  IBM watsonx.ai      │
                    │  Granite-13B-Instruct│
                    └──────────────────────┘
                                
   Agent 3 also uses:
   ┌──────────────────┐
   │    ChromaDB      │
   │  (Persistent     │
   │  Vector Store)   │
   │  + all-MiniLM-L6 │
   │  Embeddings      │
   └──────────────────┘
```

## Features

| Feature | Description |
|---|---|
| 📄 **Contract Reviewer** | Regex heuristics + LLM analysis flags HIGH/MEDIUM/LOW/INFO risk clauses |
| ⚖️ **Compliance Checker** | Cross-checks against 9 statutory rules (NLRA, FTC, GDPR, CCPA, UCTA, FAA, DTSA, UCC, IP law) |
| 🔍 **RAG Q&A Engine** | ChromaDB vector store + sentence-transformers + Granite generation for grounded answers |
| 📝 **Plain-English Summarizer** | Client-friendly summary with key points, obligations, deadlines, and legal disclaimer |
| 🟡 **Offline Mock Mode** | Full UI works without IBM credentials using a deterministic mock LLM |

---

## Project Structure

```
.
├── app.py                        # Streamlit dashboard (entry point)
├── requirements.txt              # Python dependencies
├── .env.example                  # Credential template
├── agents/
│   ├── __init__.py
│   ├── contract_reviewer.py      # Agent 1 — risky clause detection
│   ├── compliance_checker.py     # Agent 2 — statutory compliance
│   ├── qa_rag.py                 # Agent 3 — ChromaDB RAG + Q&A
│   └── summarizer.py             # Agent 4 — plain-English summary
├── utils/
│   ├── __init__.py
│   └── granite.py                # IBM Granite client + mock fallback
└── samples/
    └── sample_agreement.txt      # Sample NDA for testing
```

---

## Setup

### Prerequisites

- Python 3.10+
- IBM Cloud account with watsonx.ai project *(optional — app works offline without it)*

### 1. Clone / download this project

```bash
git clone <repo-url>
cd ai-legal-aid
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run downloads the `all-MiniLM-L6-v2` sentence-transformers model (~90 MB).

### 4. Configure credentials *(optional — for real AI responses)*

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
IBM_CLOUD_API_KEY=your_actual_api_key
IBM_CLOUD_URL=https://us-south.ml.cloud.ibm.com
IBM_PROJECT_ID=your_watsonx_project_id
```

To get these values:
1. Log in to [IBM Cloud](https://cloud.ibm.com)
2. Create or open a **watsonx.ai** project
3. Generate an API key in **Manage → Access (IAM) → API keys**
4. Copy the Project ID from your watsonx.ai project settings

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Quick Start (no credentials)

1. Run `streamlit run app.py`
2. Go to **Document Upload** → click **Load Sample NDA**
3. Switch to **Multi-Agent Analysis** → click **Run All Agents**
4. Explore results in **Risk Breakdown** and ask questions in **Legal Q&A**

All four agents run in offline mock mode — the full UI is functional, and all
heuristic-based analysis (clause detection, compliance keyword matching) runs
without any API calls.

---

## Agent Details

### Agent 1 — Contract Reviewer (`agents/contract_reviewer.py`)

- **Heuristic scan:** 10 regex patterns for known risky clause types
- **LLM deep review:** IBM Granite produces structured clause-by-clause analysis
- **Output:** `ContractReviewResult` with list of `RiskyClause` objects, each with title, excerpt, risk level (HIGH/MEDIUM/LOW/INFO), and plain-English explanation

### Agent 2 — Compliance Checker (`agents/compliance_checker.py`)

- **Static rule-base:** 9 statutory rules covering US and EU/UK law
- **Keyword trigger:** matches contract text against rule trigger keywords
- **LLM enrichment:** Granite provides context-aware analysis per triggered rule
- **Output:** `ComplianceResult` with severity-ranked `ComplianceIssue` list and overall status (NON_COMPLIANT / WARNINGS / COMPLIANT)

### Agent 3 — RAG Q&A Engine (`agents/qa_rag.py`)

- **Indexing:** Chunks document (500-char, 80-char overlap) → `all-MiniLM-L6-v2` embeddings → ChromaDB
- **Retrieval:** Cosine similarity search for top-5 relevant chunks
- **Generation:** Granite generates grounded answers from retrieved context
- **Output:** `QAResult` with answer text and source excerpts with similarity scores

### Agent 4 — Plain-English Summarizer (`agents/summarizer.py`)

- **Structured prompt:** Extracts document type, parties, dates, term, governing law
- **Key points:** Categorised as Obligation / Restriction / Right / Deadline / Risk
- **Disclaimer:** Always appends mandatory legal disclaimer
- **Output:** `SummaryResult` with all structured fields

---

## Tech Stack

| Component | Library | Version |
|---|---|---|
| UI | Streamlit | ≥ 1.35 |
| LLM | ibm-watsonx-ai | ≥ 1.1 |
| LLM Model | IBM Granite 13B Instruct | v2 |
| Orchestration | LangChain | ≥ 0.2 |
| Vector DB | ChromaDB | ≥ 0.5 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | ≥ 3.0 |
| PDF Parsing | pypdf | ≥ 4.2 |
| Env Config | python-dotenv | ≥ 1.0 |

---

## Disclaimer

> This tool is for **informational and educational purposes only**. It does not constitute legal advice. Always consult a qualified attorney before signing any legal agreement. AI-generated analysis may be incomplete or inaccurate.

---

*Built with ❤️ using IBM Granite · LangChain · ChromaDB · Streamlit*
