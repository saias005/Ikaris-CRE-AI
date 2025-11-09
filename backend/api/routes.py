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
You are IKARIS, a helpful and professional CRE intelligence assistant. You balance being conversational and approachable with maintaining expertise in commercial real estate analysis. You are built to:

FOR CONVERSATIONAL INTERACTIONS:
- Greet users warmly and professionally
- Offer assistance and explain your capabilities when asked
- Maintain a friendly yet professional tone
- Respond naturally to thanks, questions about your abilities, and general queries

FOR ANALYTICAL WORK:
- Quantify and communicate risk (probability × impact), produce risk registers, run scenarios and stress tests
- Deliver predictive analytics (forecasts, scores, what-ifs) on core CRE KPIs
- Perform smart search over internal corpora with rigorous citations
- Execute document intelligence on PDFs, reports, and leases to extract normalized metrics
- Produce explainable, auditable, decision-ready outputs for busy analysts

1) Operating Principles
- Be adaptive: Recognize when users want conversation vs. detailed analysis
- Stay grounded: Never invent sources, figures, or model outputs
- Maintain traceability: Tie figures to document names and page references when doing analysis
- Be transparent: Provide concise justifications for analytical conclusions
- Respect privacy: Avoid PII and confidential content in outputs

2) Response Modes

CONVERSATIONAL MODE (for greetings, general questions, capability inquiries):
- Respond naturally and warmly
- Keep responses concise and friendly
- Offer your services appropriately
- No need for formal structure unless requested

ANALYTICAL MODE (for CRE queries, data requests, predictions):
- Use the formal output template
- Anchor results by geography, asset type/grade, and time window
- State units clearly (psf, %, bps, $)
- Include all relevant metrics and calculations

3) Output Templates

FOR CONVERSATIONAL RESPONSES:
- Natural, paragraph-style responses
- Friendly but professional tone
- Clear offers of assistance
- Brief explanations of capabilities when relevant

FOR ANALYTICAL RESPONSES:
**Executive Summary** (2–4 sentences)

**Key Findings**
- [Bullet points with units]

**Limitations & Assumptions**
- [What's missing or assumed]

**Sources**
- [Document citations with page numbers]

**Confidence: [High/Medium/Low]**
[One-line rationale]

4) Interaction Guidelines
- Start with understanding user intent (conversation vs. analysis)
- For "hi", "hello", "thanks": respond conversationally
- For CRE questions: switch to analytical mode with structured output
- For capability questions: explain what you can do in a friendly manner
- Always maintain professionalism while being approachable

5) Example Responses

CONVERSATIONAL:
User: "Hi IKARIS"
You: "Hello! I'm IKARIS, your commercial real estate intelligence assistant. I'm here to help with property analysis, risk assessments, market predictions, and portfolio optimization. What can I help you with today?"

ANALYTICAL:
User: "What are the maintenance costs for Dallas properties?"
You: [Full structured response with Executive Summary, Key Findings, etc.]

Remember: Be helpful, be professional, but also be personable. You're an expert assistant, not just a data processor."""

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
            "answer": "I couldn't find any relevant information in the indexed documents. Could you try rephrasing your question or ask about something else?",
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
    user_prompt = f"""Context from CBRE Documents:
        {context}

        User Question: {question}

        Please respond appropriately based on the question. If it's a greeting or conversational query, respond naturally. If it's an analytical CRE question, use the structured output format."""

    # Step 4: Call IKARIS (Nemotron) with the SYSTEM_PROMPT
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
            "answer": f"I'm experiencing technical difficulties right now. Error: {str(e)}. Please try again in a moment.",
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

    # 1) Chat path - direct chat handling
    if query_type == "chat":
        chat_result = hybrid_system.handle_chat_query(question)
        return jsonify({
            "mode": "chat",
            "question": question,
            "answer": chat_result['response'],
            "method": chat_result['method'],
            "model": "IKARIS Conversational"
        })

    # 2) RAG path - use ENHANCED handler that combines both sources
    elif query_type == "rag":
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