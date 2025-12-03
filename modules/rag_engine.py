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

import config
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from modules.llm import get_llm
from langchain_neo4j import GraphCypherQAChain, Neo4jVector
from langchain_core.prompts import PromptTemplate
from modules.llm import get_llm, get_embeddings

# --- PROMPTS ---
CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to question a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

CRITICAL STRATEGY:
1. **Search Logic:** ALWAYS retrieve the node AND all its immediate relationships.
2. **Fuzzy Matching:** Always use `toLower(n.id) CONTAINS "term"`.
3. **Universal Pattern:** `MATCH (n)-[r]-(m) WHERE toLower(n.id) CONTAINS "term" RETURN n, r, m`

The question is:
{question}"""

HYBRID_QA_TEMPLATE = """You are Mimir, an advanced hybrid AI assistant.
You have context from two sources: Structured Knowledge Graph and Semantic Vector Search.

---
🔍 Vector Context (Semantic Matches):
{vector_context}
---
🕸️ Graph Context (Relationships):
{graph_context}
---

User Question:
{question}

Instructions:
1. Combine insights from both contexts.
2. If the Graph provides specific relationships, prioritize them.
3. If the Vector context provides definitions or details, include them.
4. Answer professionally.

Answer:"""

class HybridRAG:
    def __init__(self, graph_db, model_name):
        self.graph = graph_db
        self.llm = get_llm(model_name, temperature=0)
        self.embeddings = get_embeddings()

        # 1. Setup Vector Retriever (Try connecting to existing index)
        try:
            self.vector_store = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=config.NEO4J_URI,
                username=config.NEO4J_USERNAME,
                password=config.NEO4J_PASSWORD,
                index_name="vector_index",
                node_label="Chunk"
            )
        except Exception as e:
            print(f"⚠️ Vector index not found (graph empty?): {e}")
            self.vector_store = None

        # 2. Setup Graph Chain
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=CYPHER_GENERATION_TEMPLATE
        )
        self.graph_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=cypher_prompt,
            return_direct=True # Return raw data, let final LLM synthesize
        )

    def query(self, user_question):
        # A. Vector Search
        vector_context = "No vector data found."
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(user_question, k=3)
                vector_context = "\n".join([d.page_content for d in docs])
            except Exception as e:
                print(f"Vector search warning: {e}")

        # B. Graph Search
        graph_context = "No graph data found."
        try:
            graph_result = self.graph_chain.invoke({"query": user_question})
            graph_context = str(graph_result.get('result', ''))
        except Exception as e:
            print(f"Graph search warning: {e}")

        # C. Hybrid Synthesis
        final_prompt = PromptTemplate(
            input_variables=["vector_context", "graph_context", "question"],
            template=HYBRID_QA_TEMPLATE
        )

        chain = final_prompt | self.llm
        response = chain.invoke({
            "vector_context": vector_context,
            "graph_context": graph_context,
            "question": user_question
        })

        return response.content

def get_qa_chain(graph_db, model_name, verbose=True):
    """Returns the Hybrid RAG engine."""
    return HybridRAG(graph_db, model_name)