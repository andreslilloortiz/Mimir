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

from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from modules.llm import get_llm

# --- PROMPTS ---
CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to question a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

CRITICAL STRATEGY (THE "CONTEXT" RULE):
1. **Search Logic:** When the user asks about a specific Topic, Technology, or Person, DO NOT just look for the node. **ALWAYS** retrieve the node AND all its immediate relationships.
2. **Fuzzy Matching:** Always use `toLower(n.id) CONTAINS "term"` to handle spelling variations.
3. **The Universal Pattern:**
   - `MATCH (n)-[r]-(m) WHERE toLower(n.id) CONTAINS "your search term" RETURN n, r, m`

The question is:
{question}"""

QA_TEMPLATE = """You are Mimir, an expert assistant.
You have retrieved the following context from a Knowledge Graph to answer the user's question.

Context (Graph Data):
{context}

User Question:
{question}

INSTRUCTIONS FOR THE ANSWER:
1. **Analyze the Context:** Look at the relationships.
2. **Synthesize:** Don't just list facts. Combine them into coherent sentences.
3. **Be Comprehensive:** Use ALL the information provided in the context.
4. **Style:** Professional, technical, and detailed.

Answer:"""

def get_qa_chain(graph_db, verbose=True):
    """Creates the Graph RAG chain."""
    llm = get_llm(temperature=0)

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=CYPHER_GENERATION_TEMPLATE
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=QA_TEMPLATE
    )

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph_db,
        verbose=verbose,
        allow_dangerous_requests=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt
    )
    return chain