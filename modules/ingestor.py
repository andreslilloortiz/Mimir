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
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from modules.llm import get_llm

def process_pdf(pdf_path, graph_db):
    """
    Loads a PDF, extracts the graph using LLM, and stores it in Neo4j.
    Returns a dictionary with processing stats.
    """
    # 1. Load PDF
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Initialize Transformer
    # strict mode uses temperature=0 for factual extraction
    llm = get_llm(temperature=0)
    llm_transformer = LLMGraphTransformer(llm=llm)

    # 3. Extract Graph Data
    start_time = time.time()
    # NOTE: For huge PDFs, you might want to slice documents[:5] for testing
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    duration = time.time() - start_time

    # 4. Persist to Neo4j
    graph_db.add_graph_documents(graph_documents)

    return {
        "pages": len(documents),
        "entities": len(graph_documents),
        "duration": duration
    }