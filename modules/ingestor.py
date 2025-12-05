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
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader, WebBaseLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modules.llm import get_llm, get_embeddings
import config

def get_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf": return PyPDFLoader(file_path)
    elif ext == ".docx": return Docx2txtLoader(file_path)
    elif ext == ".txt": return TextLoader(file_path, encoding="utf-8")
    elif ext == ".md": return UnstructuredMarkdownLoader(file_path)
    else: raise ValueError(f"Unsupported file format: {ext}")

def _run_pipeline(documents, graph_db, model_name, source_name):
    """
    Core pipeline: Split -> Graph Extraction -> Vector Indexing.
    Used by both File and URL ingestors.
    """
    # 1. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Overwrite source metadata for better citations
    for doc in chunks:
        doc.metadata["source"] = source_name

    # 2. GRAPH EXTRACTION (Structured)
    llm = get_llm(model_name=model_name, temperature=0)
    llm_transformer = LLMGraphTransformer(llm=llm)

    start_time = time.time()
    graph_documents = llm_transformer.convert_to_graph_documents(chunks)

    if graph_documents:
        graph_db.add_graph_documents(graph_documents)

    # 3. VECTOR INDEXING (Unstructured/Semantic)
    embeddings = get_embeddings()

    Neo4jVector.from_documents(
        chunks,
        embeddings,
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        index_name="vector_index",
        node_label="Chunk"
    )

    duration = time.time() - start_time

    return {
        "pages": len(chunks),
        "entities": len(graph_documents),
        "duration": duration
    }

def process_file(file_path, graph_db, model_name, original_filename=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        loader = get_loader(file_path)
        documents = loader.load()
        return _run_pipeline(documents, graph_db, model_name, original_filename or file_path)
    except Exception as e:
        raise RuntimeError(f"Error processing file: {e}")

def process_url(url, graph_db, model_name):
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        return _run_pipeline(documents, graph_db, model_name, url)
    except Exception as e:
        raise RuntimeError(f"Error processing URL: {e}")