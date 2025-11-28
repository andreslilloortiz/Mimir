import argparse
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama
from langchain_neo4j import Neo4jGraph

# --- CONFIGURATION ---
# Connection details for the Dockerized Neo4j instance
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password123"

# Model configuration
MODEL_NAME = "llama3.2"

# --- BIG DATA & ENGINEERING ONTOLOGY ---
# We enforce a strict schema so Mimir understands the difference between
# a theoretical concept (CAP Theorem) and a concrete tool (Docker).
ALLOWED_NODES = [
    "Document",     # The PDF file itself
    "Concept",      # Theoretical concepts (e.g., CAP Theorem, ACID, Cross-compilation)
    "Technology",   # General tools/languages (e.g., Python, Docker, GCC, QEMU)
    "Database",     # Specific storage systems (e.g., Neo4j, HDFS, Cassandra, MongoDB)
    "Architecture", # System designs (e.g., Master-Slave, Peer-to-Peer, Lambda Arch)
    "Algorithm",    # Logic/Processes (e.g., MapReduce, PageRank, Ridge Regression)
    "Metric",       # Evaluation metrics (e.g., Precision, Recall, Latency, Throughput)
    "Person",       # Authors, Researchers (e.g., Andrés Lillo, Dean & Ghemawat)
    "Organization"  # Universities, Companies (e.g., USC, Televes, Google)
]

ALLOWED_RELS = [
    "AUTHOR_OF",    # Person -> Document
    "MENTIONS",     # Document -> Concept/Tech/Database...
    "DEFINES",      # Document -> Concept (Stronger than mentions)
    "USES",         # Technology -> Technology (e.g., 'Pookie uses Docker')
    "IMPLEMENTS",   # Database -> Architecture (e.g., 'HDFS implements Master-Slave')
    "EVALUATES",    # Metric -> Algorithm (e.g., 'F1-Score evaluates Classifier')
    "RELATED_TO",   # Concept -> Concept
    "TYPE_OF"       # MongoDB -> Database
]

def parse_arguments():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Mimir Ingest: Converts a PDF into a Knowledge Graph using LLMs."
    )
    # This defines the argument. It's mandatory (no '--' prefix).
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the PDF file you want to ingest (e.g., docs/paper.pdf)"
    )
    # Optional flag to clear database before running
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the database before ingesting new data"
    )
    return parser.parse_args()

def main():
    # 1. PARSE ARGUMENTS
    args = parse_arguments()
    pdf_path = args.file_path

    # 2. VALIDATION
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: File not found at '{pdf_path}'.")
        return

    print(f"🚀 Starting Mimir Ingestion Pipeline...")
    print(f"   -> Target File: {pdf_path}")
    print(f"   -> Model: {MODEL_NAME}")
    print(f"   -> Schema: {len(ALLOWED_NODES)} node types defined.")

    # 2. CONNECT TO NEO4J DATABASE
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
    # Check if the user passed the --clear flag
        if args.clear:
            print("🧹 Flag --clear detected. Wiping database...")
            graph.query("MATCH (n) DETACH DELETE n")
    except Exception as e:
        print(f"❌ Connection Failed: Could not connect to Neo4j.\nError: {e}")
        return

    # 3. LOAD AND PARSE PDF
    print(f"📄 Loading document...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   -> Successfully loaded {len(documents)} pages.")

    # 4. INITIALIZE THE LLM (BRAIN)
    # temperature=0 ensures the model is deterministic and factual (no creativity)
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
        base_url="http://localhost:11434" # Pointing to local Ollama instance (exposed by Docker)
    )

    # 5. INITIALIZE GRAPH TRANSFORMER
    # This is the 'Magic' component that converts raw text into Nodes and Relationships
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELS,
        )

    # 6. EXECUTE GRAPH EXTRACTION (ETL Process)
    print("🧠 Extracting Entities and Relationships (GraphRAG)...")
    print("   (This process is GPU-intensive, please wait...)")

    start_time = time.time()

    # NOTE: If your PDF is huge, you might want to process only the first few pages for testing:
    # graph_documents = llm_transformer.convert_to_graph_documents(documents[:5])
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    end_time = time.time()
    print(f"   -> Processing complete in {end_time - start_time:.2f} seconds.")
    print(f"   -> Extracted {len(graph_documents)} graph structures.")

    # 7. LOAD DATA INTO NEO4J
    print("💾 Persisting data to Neo4j Graph Database...")
    graph.add_graph_documents(graph_documents)

    print("✅ INGESTION FINISHED SUCCESSFULLY!")
    print("\n--- NEXT STEPS ---")
    print("1. Open Neo4j Browser: http://localhost:7474")
    print("2. Login with: neo4j / password123")
    print("3. Visualize your data with this query:")
    print("   MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50")

if __name__ == "__main__":
    main()