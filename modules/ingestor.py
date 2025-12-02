# This file is part of Mimir.

# Copyright (C) 2025 Andrés Lillo Ortiz

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import time
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modules.llm import get_llm, get_embeddings

def get_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf": return PyPDFLoader(file_path)
    elif ext == ".docx": return Docx2txtLoader(file_path)
    elif ext == ".txt": return TextLoader(file_path, encoding="utf-8")
    elif ext == ".md": return UnstructuredMarkdownLoader(file_path)
    else: raise ValueError(f"Unsupported file format: {ext}")

def process_file(file_path, graph_db, model_name):
    """
    Ingests data using HYBRID approach:
    1. Knowledge Graph Extraction (LLMGraphTransformer)
    2. Vector Indexing (Neo4jVector)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load Content
    try:
        loader = get_loader(file_path)
        raw_documents = loader.load()
    except Exception as e:
        raise RuntimeError(f"Error loading document: {e}")

    # 2. Split Text (Crucial for Vector Search)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_documents)

    # 3. GRAPH EXTRACTION (Structured)
    # Smaller models are faster for this step
    llm = get_llm(model_name=model_name, temperature=0)
    llm_transformer = LLMGraphTransformer(llm=llm)

    print("🕸️ Extracting Graph Data...")
    start_time = time.time()
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    if graph_documents:
        graph_db.add_graph_documents(graph_documents)

    # 4. VECTOR INDEXING (Unstructured/Semantic)
    print("🔢 Generating Vectors...")
    embeddings = get_embeddings()

    # Creates a vector index in Neo4j named "vector_index"
    Neo4jVector.from_documents(
        documents,
        embeddings,
        url=graph_db.url,
        username=graph_db.username,
        password=graph_db.password,
        index_name="vector_index",
        node_label="Chunk"
    )

    duration = time.time() - start_time

    return {
        "pages": len(documents),
        "entities": len(graph_documents),
        "duration": duration
    }