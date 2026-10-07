import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# Load environment configuration
load_dotenv()

# Streamlit UI Setup
st.set_page_config(
    page_title="Dubai Farms Knowledge Graph RAG",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Dubai Farms Programme — Graph RAG & Evaluation")
st.caption(
    "Powered by Neo4j, LangChain GraphTransformer, Guardrails, and DeepEval"
)

# Target UAE Website URL
TARGET_URL = "https://u.ae/en/about-the-uae/strategies-initiatives-and-awards/strategies-plans-and-visions/environment-and-energy/the-dubai-farms-programme"


# =====================================================================
# STEP 1: DATABASE & MODEL INITIALIZATION
# =====================================================================
@st.cache_resource
def get_graph_connection():
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )


llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")


# =====================================================================
# STEP 2: WEBPAGE SCRAPING & GRAPH INGESTION
# =====================================================================
def ingest_website_to_graph():
    graph = get_graph_connection()

    # Clear previous graph state
    graph.query("MATCH (n) DETACH DELETE n")

    # Scrape Web Page Content
    loader = WebBaseLoader(TARGET_URL)
    documents = loader.load()

    # Extract Graph Entities & Relationships
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # Store in Neo4j
    graph.add_graph_documents(graph_documents)
    graph.refresh_schema()


# Sidebar button for Graph Ingestion
with st.sidebar:
    st.header("⚙️ Graph Operations")
    if st.button("📥 Ingest Dubai Farms Webpage"):
        with st.spinner("Scraping webpage & constructing Knowledge Graph..."):
            try:
                ingest_website_to_graph()
                st.success("Successfully built and loaded graph into Neo4j!")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")


# =====================================================================
# STEP 3: GUARDRAILS CHECK
# =====================================================================
def apply_guardrails(query: str) -> tuple[bool, str]:
    """Validates query safety using keyword and domain boundary checks."""
    blocked_keywords = [
        "hack",
        "bomb",
        "exploit",
        "malware",
        "bypass",
        "election",
        "politics",
    ]

    for term in blocked_keywords:
        if re.search(r"\b" + re.escape(term) + r"\b", query, re.IGNORECASE):
            return (
                False,
                f"Query rejected: Safety policy violation detected ('{term}').",
            )

    return True, "Passed"


# =====================================================================
# STEP 4: GRAPH FACT RETRIEVAL
# =====================================================================
def retrieve_graph_facts(question: str) -> list[str]:
    """Retrieves 2-hop connected entity paths from Neo4j."""
    graph = get_graph_connection()

    stopwords = {
        "what",
        "who",
        "where",
        "how",
        "does",
        "the",
        "and",
        "is",
        "are",
        "tell",
        "me",
        "about",
    }
    keywords = [
        w.strip("?,.!")
        for w in question.lower().split()
        if len(w) > 2 and w not in stopwords
    ]

    cypher_query = """
    MATCH (n)
    WHERE toLower(n.id) CONTAINS toLower($term)
    OPTIONAL MATCH path = (n)-[*1..2]-(m)
    WITH DISTINCT path
    RETURN [rel IN relationships(path) | 
            startNode(rel).id + ' -[' + type(rel) + ']-> ' + endNode(rel).id
           ] AS triples
    """

    retrieved_facts = set()
    for term in keywords:
        results = graph.query(cypher_query, {"term": term})
        for record in results:
            if record["triples"]:
                for triple in record["triples"]:
                    retrieved_facts.add(triple)

    # Fallback if specific keywords do not match
    if not retrieved_facts:
        all_facts = graph.query(
            "MATCH (a)-[r]->(b) RETURN a.id + ' -[' + type(r) + ']-> ' + b.id AS triple LIMIT 30"
        )
        retrieved_facts = {row["triple"] for row in all_facts}

    return list(retrieved_facts)


# =====================================================================
# STEP 5: DEEPEVAL EVALUATION
# =====================================================================
def run_deepeval_metrics(
    question: str, answer: str, context: list[str]
) -> dict:
    """Evaluates generator groundedness and relevancy."""
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context
        if context
        else ["No facts retrieved from graph."],
    )

    faithfulness = FaithfulnessMetric(threshold=0.7)
    relevancy = AnswerRelevancyMetric(threshold=0.7)

    faithfulness.measure(test_case)
    relevancy.measure(test_case)

    return {
        "faithfulness_score": faithfulness.score,
        "faithfulness_reason": faithfulness.reason,
        "relevancy_score": relevancy.score,
        "relevancy_reason": relevancy.reason,
    }


# =====================================================================
# STEP 6: STREAMLIT CHAT INTERFACE & GENERATION
# =====================================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display prior conversation history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input(
    "Ask a question about the Dubai Farms Programme..."
)

if user_input:
    # Append user question
    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # 1. Check Guardrails
        is_safe, guardrail_msg = apply_guardrails(user_input)

        if not is_safe:
            st.warning(f"🛡️ Guardrail Triggered: {guardrail_msg}")
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": f"🛡️ Guardrail Triggered: {guardrail_msg}",
                }
            )
        else:
            # 2. Retrieve Facts
            with st.spinner("Searching Neo4j Knowledge Graph..."):
                facts = retrieve_graph_facts(user_input)

            # Context Display
            with st.expander("🕸️ View Retrieved Graph Context", expanded=False):
                if facts:
                    for f in facts:
                        st.code(f, language="text")
                else:
                    st.write("No matching graph paths found.")

            # 3. Generate Answer
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an assistant for the UAE Government portal. "
                        "Answer strictly using the provided Knowledge Graph facts. "
                        "If facts do not answer the prompt, say so.\n\nFacts:\n{context}",
                    ),
                    ("human", "{question}"),
                ]
            )

            context_str = "\n".join(facts)
            chain = qa_prompt | llm
            response = chain.invoke(
                {"context": context_str, "question": user_input}
            )
            answer_text = response.content

            st.markdown(answer_text)

            # 4. DeepEval Quality Check
            with st.spinner("Evaluating response metrics with DeepEval..."):
                try:
                    eval_results = run_deepeval_metrics(
                        user_input, answer_text, facts
                    )

                    with st.expander(
                        "📊 Response Quality Evaluation (DeepEval)", expanded=True
                    ):
                        col1, col2 = st.columns(2)
                        col1.metric(
                            "Faithfulness",
                            f"{eval_results['faithfulness_score']:.2f}",
                        )
                        col2.metric(
                            "Answer Relevancy",
                            f"{eval_results['relevancy_score']:.2f}",
                        )

                        st.caption(
                            f"**Faithfulness Reason:** {eval_results['faithfulness_reason']}"
                        )
                        st.caption(
                            f"**Relevancy Reason:** {eval_results['relevancy_reason']}"
                        )
                except Exception as e:
                    st.info(f"Evaluation note: {e}")

            # Store response in session memory
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer_text}
            )