# 🌾 Dubai Farms Programme — Graph RAG & Evaluation Chatbot

An end-to-end **Graph Retrieval-Augmented Generation (Graph RAG)** system built with **Streamlit**, **Neo4j**, **LangChain**, and **DeepEval**. 

Unlike conventional vector-based RAG pipelines that depend solely on semantic similarity search, this application parses raw web content into an interconnected knowledge graph. By retrieving multi-hop relationships directly from Neo4j, the chatbot answers queries about the UAE's **Dubai Farms Programme** with factual accuracy, verified grounding, and automated evaluation metrics.

---

## 🌟 Key Features

- **🌐 Direct Web Ingestion:** Automatically scrapes the live UAE government portal page using LangChain's `WebBaseLoader`.
- **🕸️ Automated Knowledge Graph Generation:** Uses `LLMGraphTransformer` to extract structured entities and semantic relationships (`(Subject)-[RELATION]->(Object)`), storing them directly into Neo4j.
- **🛡️ Input Guardrails:** Pre-screens prompts with regular expressions to filter malicious terms, safety violations, and off-topic domain queries before processing.
- **🔍 2-Hop Graph Traversal:** Extracts connected context paths (up to 2 relationship hops) using Cypher queries based on extracted query keywords.
- **🤖 Grounded Generation:** Prompts `gpt-4o-mini` with a strict zero-hallucination constraint based only on retrieved knowledge graph facts.
- **📊 Real-time LLM Evaluation:** Employs **DeepEval** to compute and display real-time **Faithfulness** and **Answer Relevancy** scores along with natural language reasoning.

---

## 🏗️ Architecture Flow

```
[ UAE Government Portal ]
           │
           ▼ (LangChain WebBaseLoader)
     [ Raw HTML ]
           │
           ▼ (LLMGraphTransformer / gpt-4o-mini)
 [ Entity-Relationship Triples ]
           │
           ▼
    [ Neo4j Database ]
           ▲
           │ (1-2 Hop Cypher Query)
    [ Graph Retrieval ] ◄────────────── [ User Query ]
           │                                    │
           ▼                                    ▼
[ Grounded Context Prompt ]            [ Safety Guardrails ]
           │
           ▼ (gpt-4o-mini)
   [ Final Response ]
           │
           ▼
[ DeepEval Audit (Faithfulness & Relevancy) ]
```

---

## 🛠️ Tech Stack

- **Frontend / Framework:** [Streamlit](https://streamlit.io/)
- **Orchestration:** [LangChain](https://www.langchain.com/) / `langchain-neo4j` / `langchain-openai`
- **Knowledge Graph Database:** [Neo4j](https://neo4j.com/)
- **Foundation LLM:** OpenAI `gpt-4o-mini`
- **Evaluation Framework:** [DeepEval](https://github.com/confident-ai/deepeval)
- **Environment Management:** `python-dotenv`

---

## 📋 Prerequisites

Ensure you have the following ready before starting:

1. **Python 3.10+** installed on your system.
2. A running **Neo4j instance** (Neo4j Desktop, self-hosted Docker container, or a cloud-hosted [Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/) instance).
3. An **OpenAI API Key** with access to `gpt-4o-mini`.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/dubai-farms-graph-rag.git
cd dubai-farms-graph-rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

*(If you don't have a `requirements.txt` yet, install the core packages directly:)*

```bash
pip install streamlit langchain langchain-community langchain-core \
            langchain-experimental langchain-neo4j langchain-openai \
            deepeval python-dotenv neo4j
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Connection Details
NEO4J_URI=bolt://localhost:7687       # Or your neo4j+s:// URL for AuraDB
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 📖 How to Use the Demo

1. **Ingest the Knowledge Graph:**
   - On the left sidebar, click **"📥 Ingest Dubai Farms Webpage"**.
   - The application will scrape the target UAE portal page, extract graph structures, and store nodes and relationships into Neo4j.
2. **Ask Questions:**
   - In the chat bar, ask domain-specific queries such as:
     - *"What is the main objective of the Dubai Farms Programme?"*
     - *"Who does the programme support?"*
3. **Inspect Facts & Evaluation:**
   - Expand the **"🕸️ View Retrieved Graph Context"** tab to see exact graph triples queried from Neo4j.
   - Inspect the **"📊 Response Quality Evaluation (DeepEval)"** card to view real-time **Faithfulness** and **Relevancy** metric scores.

---

## 🔒 Safety & Guardrails

The application intercepts queries containing flagged keywords (e.g., system manipulation, malicious commands, political/destructive content). When triggered, the system rejects processing before issuing any database queries or LLM calls.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).