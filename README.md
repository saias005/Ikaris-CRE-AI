# IKARIS — CRE Intelligence OS

> Built at **HackUTD 2025** in 24 hours. AI-powered commercial real estate assistant for CBRE, combining a hybrid RAG pipeline with 5 trained ML models to answer analytical questions about a property portfolio.

**Team:** Sai Abishek Srinivasan · Rohan Patel · [teammates]  
**Hackathon:** HackUTD 2025 — NVIDIA / CBRE Challenge Track  
**My Role:** ML pipeline (model training, query routing, prediction handlers), hybrid RAG integration

---

## What It Does

IKARIS answers questions a CBRE analyst would actually ask — not just "find documents about Dallas" but "which properties in Dallas are at risk of lease non-renewal, and what's the predicted maintenance cost increase over the next year?"

It routes each query to the right system:

| Query Type | Example | Handler |
|---|---|---|
| **Factual** | "Show me Class A office buildings in Houston" | RAG (ChromaDB + CSV filter) |
| **Prediction** | "Forecast maintenance costs for next quarter" | ML — Gradient Boosting |
| **Risk** | "Which properties are at high lease risk?" | ML — Random Forest Classifier |
| **Optimization** | "Find undervalued properties in our portfolio" | ML — Random Forest Regressor |
| **Conversational** | "What can you help me with?" | Rule-based chat handler |

---

## ML Models

All 5 models are trained at startup on the CBRE portfolio CSV data:

| Model | Algorithm | Target | Features |
|---|---|---|---|
| **Property Valuation** | Random Forest Regressor | Property value | sqft, age, occupancy, NOI, cap rate, market |
| **Maintenance Forecasting** | Gradient Boosting Regressor | Future maintenance cost | Building age, sqft, risk score, historical maintenance |
| **Lease Risk** | Random Forest Classifier | High renewal risk (binary) | WALT, occupancy, tenant risk score, payment history |
| **Energy Efficiency** | Random Forest Classifier | High energy cost flag | Energy Star score, cost/sqft, building age, LEED status |
| **Occupancy Prediction** | Random Forest Regressor | Future occupancy rate | Current occupancy, market vacancy, WALT, base rent |

---

## Architecture

```
User Query
    ↓
Query Classifier (keyword routing)
    ├── "predict / forecast"  → ML Prediction Handler
    ├── "risk / vulnerable"   → ML Risk Handler
    ├── "optimize / best"     → ML Optimization Handler
    ├── "hi / hello"          → Chat Handler
    └── default               → RAG Handler
                                    ├── ChromaDB (CBRE PDFs, semantic search)
                                    └── CSV filter (structured property data)
```

**RAG Pipeline:**
- 15+ real CBRE market reports and white papers ingested as PDFs
- HuggingFace `all-MiniLM-L6-v2` for local embeddings (no API key needed)
- ChromaDB as the vector store
- NVIDIA NeMo (`meta/llama-3.1-70b-instruct`) as the LLM
- Dynamic query filters applied on CSV data before combining with semantic results

---

## Tech Stack

**Backend:** Python, Flask, LangChain, ChromaDB, scikit-learn, pandas, OpenAI SDK (NVIDIA-compatible)

**Frontend:** React, JavaScript, Tailwind CSS

**LLM:** NVIDIA NeMo (`meta/llama-3.1-70b-instruct`) via NVIDIA API

**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally)

**Data:** Real CBRE market reports (PDFs) + synthetic CBRE-style portfolio CSVs (properties, tenants, historical metrics, comparables)

---

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Free NVIDIA API key: [build.nvidia.com](https://build.nvidia.com/)

### Backend

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp backend/.env.example backend/.env
# Edit .env and add your NVIDIA_API_KEY

# 3. Start the backend (builds ChromaDB and trains ML models on first run)
cd backend
python api/routes.py
```

Backend runs at `http://localhost:5000`

> **First run note:** ChromaDB will be rebuilt from the PDFs (~30–60 seconds). ML models train on the CSV data (~10 seconds). Subsequent runs are faster.

### Frontend

```bash
cd frontend/ikaris-chatbot
npm install
npm start
```

Frontend runs at `http://localhost:3000`

---

## Example Queries

```
"Which properties in Dallas have the highest predicted maintenance costs next year?"
"Find undervalued office buildings in Houston"
"What are the lease renewal risks in our Class A portfolio?"
"Show me properties with high energy costs that are not LEED certified"
"Predict occupancy rates for our Chicago properties"
```

---

## Data

| File | Description |
|---|---|
| `backend/data/cbre_data/properties.csv` | 100 synthetic CBRE-style commercial properties |
| `backend/data/cbre_data/tenants.csv` | Tenant records with payment history and renewal probability |
| `backend/data/cbre_data/historical_metrics.csv` | 12-month historical NOI, occupancy, energy data |
| `backend/data/cbre_data/comparables.csv` | Comparable market transactions |
| `backend/data/pdfs/` | 15 real CBRE market reports and viewpoints |

---

## Hackathon Context

Built in 24 hours at HackUTD 2025 for the NVIDIA / CBRE challenge track. The prompt asked teams to build an AI tool that could assist CBRE analysts with real estate decision-making using NVIDIA's NeMo platform.

**Key design choices made under time pressure:**
- Used NVIDIA's OpenAI-compatible API so we could use the standard OpenAI SDK without new integrations
- Trained all ML models from scratch at startup rather than loading pre-trained models (ensures reproducibility)
- Built keyword-based query routing instead of an LLM classifier (faster, more predictable under hackathon conditions)

---

## License

MIT
