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
from modules.llm import get_llm

def get_loader(file_path):
    """Factory method to choose the right loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    elif ext == ".md":
        return UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def process_file(file_path, graph_db):
    """
    Loads a file (PDF, DOCX, TXT, MD), extracts graph data, and stores it in Neo4j.
    """
    # 1. Select Loader and Load Content
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        loader = get_loader(file_path)
        documents = loader.load()
    except Exception as e:
        raise RuntimeError(f"Error loading document: {e}")

    # 2. Initialize Transformer
    # strict mode uses temperature=0 for factual extraction
    llm = get_llm(temperature=0)
    llm_transformer = LLMGraphTransformer(llm=llm)

    # 3. Extract Graph Data
    start_time = time.time()
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    duration = time.time() - start_time

    # 4. Persist to Neo4j
    if graph_documents:
        graph_db.add_graph_documents(graph_documents)

    return {
        "pages": len(documents),
        "entities": len(graph_documents),
        "duration": duration
    }