# Mimir

Experimental GraphRAG Pipeline for Knowledge Graph Construction and Querying with Local LLMs.

## About the Project

Mimir is an open-source tool designed to experiment with Graph Retrieval-Augmented Generation (GraphRAG) using local Large Language Models. The application ingests PDF documents and utilizes Llama 3.2 (via Ollama) to extract entities and relationships, constructing a Knowledge Graph within a Neo4j database. For retrieval, it attempts to translate natural language questions into Cypher queries to fetch relevant context from the graph structure. This project serves as a proof-of-concept for building privacy-focused, local RAG pipelines that leverage graph structures rather than just vector embeddings.

The project is named after Mimir, the figure from Norse mythology known for his knowledge and wisdom.

## Prerequisites

Before getting started, ensure you have the following installed and configured on your system:

1.  **Docker**:

    Install Docker by following the official installation guide for your operating system: [Docker Installation Guide](https://docs.docker.com/get-docker/).

2.  **NVIDIA Container Toolkit (Optional but Recommended)**:

    Graph extraction is a computationally intensive task. To enable GPU acceleration for Ollama within Docker, you must install the NVIDIA Container Toolkit.

    Add the package repositories:

    ```bash
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```

    Install the toolkit:

    ```bash
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    ```

    Configure the Docker runtime:

    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

3.  **Python 3.10+**:

    Ensure you have a valid Python installation to run the client scripts.

## Quick Start

1.  **Set up the Python Environment**:

    Create a virtual environment and install the required dependencies (LangChain, Neo4j, Ollama, etc.).

    ```bash
    python3 -m venv mimir-venv
    source mimir-venv/bin/activate
    pip install langchain langchain-community langchain-experimental langchain_ollama langchain_neo4j neo4j ollama pypdf tiktoken
    ```

2.  **Launch the Infrastructure**:

    You can run Mimir using CPU only or with NVIDIA GPU support.

      * **Option A: CPU Only**

        ```bash
        docker compose up -d
        ```

      * **Option B: NVIDIA GPU (Recommended)**

        ```bash
        docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d
        ```

3.  **Verify Model Download**:

    The `mimir-init` container will automatically download the `llama3.2` model. You can monitor the progress with:

    ```bash
    docker logs -f mimir-init
    ```

    Wait until you see "✅ ¡llama3.2 ready\!" before proceeding.

4.  **Ingest Data**:

    Run the ingestion script to parse a PDF and populate the graph database.

    ```bash
    python3 ingest.py docs/document.pdf --clear
    ```

5.  **Start Chatting**:

    Launch the chat interface to query your knowledge graph.

    ```bash
    python3 chat.py --verbose
    ```

## Running Mimir: Command Line Options

### Ingestion (`ingest.py`)

This script handles the ETL process: loading the PDF, extracting nodes/relationships using the LLM, and persisting them to Neo4j.

| Argument | Description |
|-|-|
| `file_path` | **Mandatory**. Path to the PDF file you want to ingest (e.g., `docs/paper.pdf`). |
| `-h, --help` | Show the help message and exit. |
| `--clear` | Clear the entire Neo4j database (`MATCH (n) DETACH DELETE n`) before ingesting new data. |

### Chat Interface (`chat.py`)

This script initializes the `GraphCypherQAChain`. It translates natural language questions into Cypher queries, executes them against the database, and synthesizes the answer.

| Argument | Description |
|-|-|
| `-h, --help` | Show the help message and exit. |
| `--verbose` | Show the generated Cypher queries and internal reasoning of the chain in the console. |

## Project Structure

The repository is organized as follows:

```text
mimir/
├── docker-compose.yml         # Base Docker services (Neo4j, Ollama, Init)
├── docker-compose.nvidia.yml  # Override configuration for NVIDIA GPU support
├── ingest.py                  # Script for PDF ingestion and Graph Extraction
├── chat.py                    # Script for RAG Chat interface
├── LICENSE                                    # Project license
└── README.md                  # Project documentation
```

## Developer Notes

### Performance

The `LLMGraphTransformer` process in `ingest.py` is computationally expensive.

  * **CPU Mode**: Processing a large PDF may take significant time.
  * **GPU Mode**: Strongly recommended. Ensure you use the `docker-compose.nvidia.yml` override.

### Visualization

You can visually inspect the generated Knowledge Graph by accessing the Neo4j Browser:

  * **URL**: [http://localhost:7474](http://localhost:7474)
  * **User**: `neo4j`
  * **Password**: `password123`

Sample query to visualize the whole graph:

```cypher
MATCH (n)-[r]->(m) RETURN n,r,m
```

## References

The resources (PDF papers) used for testing and validating the functionality of this project were sourced from the following repository:

[1] T. Tharmarajasingam, thuva4/Bigdata-Papers-Reading. (Nov. 10, 2025). Accessed: Nov. 29, 2025. [Online]. Available: https://github.com/thuva4/Bigdata-Papers-Reading

## License

This project is open source and available under the terms of the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).
See the [LICENSE](LICENSE) file for the full text.

---

I hope this guide has been helpful!