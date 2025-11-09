from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv

# Fix the import path - add backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

# Now import hybrid_sys from agents (not backend.agents)
from agents.hybrid_sys import IkarisHybridSystem

load_dotenv()

app = Flask(__name__)
CORS(app)

print("🚀 Loading IKARIS - CRE Intelligence OS...")

# ==================== IKARIS SYSTEM PROMPT ====================

IKARIS_SYSTEM_PROMPT = """IKARIS — CRE Intelligence OS
Risk • Predictive • Smart Search • Document Intelligence • Agentic Orchestration

0) Identity & Mission
You are Ikaris, a commercial real estate (CRE) analyst agent working with CBRE built to:
- Quantify and communicate risk (probability × impact), produce risk registers, run scenarios and stress tests.
- Deliver predictive analytics (forecasts, scores, what-ifs) on core CRE KPIs.
- Perform smart search over internal corpora and reputable external sources with rigorous citations.
- Execute document intelligence on PDFs, decks, reports, and leases to extract normalized, traceable metrics.
- Produce explainable, auditable, decision-ready outputs for busy analysts.
- If essential information is missing, state what's missing, propose minimal assumptions, or proceed with clearly labeled assumptions.

1) Operating Principles
- Grounded over glib: never invent sources, figures, or model outputs.
- Traceability: every extracted or quoted figure is tied to a document name and page/slide reference when applicable.
- Determinism: follow explicit steps so the same inputs yield the same outputs.
- Reasoning transparency: provide concise justifications, not step-by-step private thoughts.
- Privacy & safety: avoid PII and confidential content in outputs.

2) CRE Domain Guardrails & Conventions
- Anchor all results by geography, asset type/grade, and time window (e.g., "DFW | Office Class A | 2022–2025").
- State units clearly (psf, %, bps, $) and specify currency basis.
- Normalize cross-source metrics and note the basis used.
- Lease concepts to surface when present: start/end, rollover windows, options, escalators/indexation, CAM/OPEX, TI/LC.
- Sustainability metrics (if available): EUI, energy per square foot, water per square foot, Scope 1/2.

3) Output Template (Use this every time)
**Executive Summary** (2–4 sentences)

**Key Findings**
- [Bullet points with units]

**Limitations & Assumptions**
- [What's missing or assumed]

**Sources**
- [Document citations with page numbers]

**Confidence: [High/Medium/Low]**
[One-line rationale]

Keep prose crisp; put numbers in tables when useful; focus on actionable insights."""

# ==================== SETUP VECTOR DATABASE ====================
print("📚 Loading vector database...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vectorstore = Chroma(
    persist_directory="./backend/chroma_db",
    embedding_function=embeddings
)

print(f"✅ Vector database loaded ({vectorstore._collection.count()} chunks)")

# ==================== SETUP NEMOTRON ====================
print("🤖 Connecting to Mistral-Nemotron...")

nemotron_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

print("🧠 Initializing IKARIS Hybrid ML system...")
hybrid_system = IkarisHybridSystem()
print("✅ Hybrid system ready!")

print("✅ IKARIS (Nemotron) ready!")

# ==================== HELPER FUNCTIONS ====================

def polish_with_ikaris(question: str, raw_analysis: str) -> str:
    """
    Use Nemotron to rewrite/generate a well-structured answer from
    an internal analysis string (e.g., ML output).
    """
    user_prompt = f"""You are IKARIS, a CRE analyst.

    User question:
    {question}

    Internal analysis (from IKARIS tools/ML models):
    {raw_analysis}

    Rewrite this into a clear, decision-ready answer following the IKARIS output template:
    - Executive Summary
    - Key Findings (with numbers/units)
    - Limitations & Assumptions
    - Sources (you can say 'Internal IKARIS model outputs' if there are no documents)
    - Confidence
    """

    resp = nemotron_client.chat.completions.create(
        model="mistralai/mistral-nemotron",
        messages=[
            {"role": "system", "content": IKARIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )

    return resp.choices[0].message.content

# ==================== RAG FUNCTION ====================

def query_with_ikaris(question: str, k: int = 5):
    """
    IKARIS RAG pipeline:
    1. Search vector DB for relevant chunks
    2. Build context from CBRE documents
    3. Send context + question to IKARIS (Nemotron)
    4. Return structured answer with sources
    """
    
    # Step 1: Retrieve relevant chunks
    print(f"\n🔍 IKARIS searching for: {question}")
    relevant_docs = vectorstore.similarity_search(question, k=k)
    
    if not relevant_docs:
        return {
            "answer": "**Executive Summary**\n\nNo relevant information found in the indexed documents.\n\n**Confidence: Low**\nNo matching content in current document corpus.",
            "sources": []
        }
    
    # Step 2: Build context from CBRE documents
    context = "# CBRE Document Context\n\n"
    sources_list = []
    
    for i, doc in enumerate(relevant_docs):
        source = doc.metadata.get('source', 'Unknown')
        chunk_id = doc.metadata.get('chunk_id', 'N/A')
        
        context += f"## Document {i+1}: {source} (Chunk {chunk_id})\n"
        context += doc.page_content
        context += "\n\n---\n\n"
        
        sources_list.append({
            "source": source,
            "chunk_id": chunk_id,
            "preview": doc.page_content[:200] + "..."
        })
    
    print(f"✅ Found {len(relevant_docs)} relevant chunks")
    print(f"📄 Context length: {len(context)} characters")
    
    # Step 3: Create user prompt
    user_prompt = f"""**Context from CBRE Documents:**

    {context}

    **User Question:** {question}

    Please analyze the provided CBRE documents and answer the question following the IKARIS output template. Include specific data points with units (psf, %, $) and cite sources with document names."""

    # Step 4: Call IKARIS (Nemotron)
    print("🤖 IKARIS generating response...")
    
    try:
        response = nemotron_client.chat.completions.create(
            model="mistralai/mistral-nemotron",
            messages=[
                {"role": "system", "content": IKARIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content
        
        print("✅ IKARIS response generated")
        
        # Step 5: Format response
        result = {
            "answer": answer,
            "sources": sources_list,
            "context_chunks": len(relevant_docs),
            "model": "IKARIS (Mistral-Nemotron)"
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Error calling IKARIS: {e}")
        return {
            "answer": f"**Executive Summary**\n\nError generating response: {str(e)}\n\n**Confidence: N/A**\nSystem error occurred.",
            "sources": []
        }


# ==================== API ROUTES ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "system": "IKARIS - CRE Intelligence OS",
        "documents_indexed": vectorstore._collection.count(),
        "model": "mistralai/mistral-nemotron",
        "capabilities": [
            "Smart Search",
            "Document Intelligence",
            "Risk Analysis",
            "Predictive Analytics"
        ]
    })


@app.route('/api/query', methods=['POST'])
def query_rag():
    """
    IKARIS Query Endpoint
    
    Request body:
    {
        "question": "What are the energy cost trends in Dallas properties?",
        "k": 5  // optional, number of chunks to retrieve (default: 5)
    }
    
    Response:
    {
        "question": "...",
        "answer": "... (IKARIS formatted response)",
        "sources": [...],
        "context_chunks": 5,
        "model": "IKARIS (Mistral-Nemotron)"
    }
    """
    data = request.json
    question = data.get('question')
    k = data.get('k', 5)
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    if not (1 <= k <= 10):
        return jsonify({"error": "k must be between 1 and 10"}), 400
    
    print(f"\n📝 New Query: {question}")
    
    try:
        result = query_with_ikaris(question, k=k)
        
        response = {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "context_chunks": result.get("context_chunks", 0),
            "model": result.get("model", "IKARIS")
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            "error": str(e),
            "question": question
        }), 500


@app.route('/api/search', methods=['POST'])
def search_documents():
    """
    Direct document search (no IKARIS processing)
    Returns raw chunks for inspection
    
    Request body:
    {
        "query": "Dallas retail vacancy",
        "k": 5  // optional
    }
    """
    data = request.json
    query = data.get('query')
    k = data.get('k', 5)
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        results = vectorstore.similarity_search(query, k=k)
        
        response = {
            "query": query,
            "results": [
                {
                    "source": doc.metadata.get("source"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "content": doc.page_content,
                    "preview": doc.page_content[:300] + "..."
                }
                for doc in results
            ],
            "total": len(results)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all indexed documents"""
    try:
        # Get all unique source documents
        all_docs = vectorstore.get()
        sources = set()
        
        # Check if all_docs exists and has the expected structure
        if all_docs and isinstance(all_docs, dict) and 'metadatas' in all_docs:
            metadatas = all_docs.get('metadatas', [])
            if metadatas:
                for metadata in metadatas:
                    if metadata and isinstance(metadata, dict):
                        source = metadata.get('source')
                        if source:
                            sources.add(source)
        
        return jsonify({
            "total_chunks": vectorstore._collection.count(),
            "unique_documents": len(sources),
            "documents": sorted(list(sources))
        })
    
    except Exception as e:
        print(f"❌ Error listing documents: {e}")
        return jsonify({
            "error": str(e),
            "total_chunks": 0,
            "unique_documents": 0,
            "documents": []
        }), 500


@app.route('/api/hybrid_query', methods=['POST'])
def hybrid_query():
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    print(f"\n📝 Hybrid Query: {question}")

    # 1) Let IKARIS hybrid system classify the query
    query_type = hybrid_system.classify_query(question)
    print(f"🔀 Routed as: {query_type}")

    # 2) RAG path - use ENHANCED handler that combines both sources
    if query_type == "rag":
        # Use the hybrid system's RAG handler which includes CSV data
        enhanced_result = hybrid_system.handle_rag_query(question, vectorstore=vectorstore)
        
        # Send the combined context to Nemotron for polishing
        polished = polish_with_ikaris(question, enhanced_result['response'])
        
        return jsonify({
            "mode": "rag_enhanced",
            "question": question,
            "answer": polished,
            "raw_data": enhanced_result['response'],  # For debugging
            "properties_found": enhanced_result['num_results'],
            "method": enhanced_result['method'],
            "model": "IKARIS (Enhanced RAG - PDFs + Properties)"
        })

    # 3) ML path remains the same
    else:
        ml_result = hybrid_system.process_query(question)
        raw_text = ml_result["response"]
        method = ml_result.get("method", "")
        mtype = ml_result.get("type", "")
        
        polished = polish_with_ikaris(question, raw_text)
        
        return jsonify({
            "mode": "ml",
            "question": question,
            "answer": polished,
            "raw_ml_response": raw_text,
            "ml_type": mtype,
            "ml_method": method,
            "model": "IKARIS (Hybrid ML + Mistral-Nemotron)"
        })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏢 IKARIS - CRE Intelligence OS")
    print("   Commercial Real Estate Analysis with Mistral-Nemotron")
    print("="*70)
    print("📍 API Server: http://localhost:5000")
    print("\n📖 Endpoints:")
    print("   • GET  /api/health        - System health check")
    print("   • POST /api/query         - Ask IKARIS a question (RAG)")
    print("   • POST /api/hybrid_query  - Hybrid RAG + ML query")
    print("   • POST /api/search        - Search documents (no LLM)")
    print("   • GET  /api/documents     - List all indexed documents")
    print("="*70)
    print(f"📊 Status: {vectorstore._collection.count()} chunks indexed and ready")
    print("🤖 Model: Mistral-Nemotron (128K context)")
    print("="*70)
    print("\n✅ IKARIS is online and ready for queries!\n")
    
    app.run(debug=True, port=5001, host='0.0.0.0')