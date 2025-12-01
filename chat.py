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


import argparse
from langchain_ollama import ChatOllama
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

# --- CONFIGURATION ---
# Connection details for the Dockerized Neo4j instance
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password123"

# Model configuration
MODEL_NAME = "llama3.2"

# --- CYPHER GENERATION TEMPLATE ---
CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to question a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

CRITICAL STRATEGY (THE "CONTEXT" RULE):
1. **Search Logic:** When the user asks about a specific Topic, Technology, or Person, DO NOT just look for the node. **ALWAYS** retrieve the node AND all its immediate relationships.
   - This provides the context needed to answer "What is...", "Who created...", "How is it used...".

2. **Fuzzy Matching:** Always use `toLower(n.id) CONTAINS "term"` to handle spelling variations.

3. **The Universal Pattern:**
   - `MATCH (n)-[r]-(m) WHERE toLower(n.id) CONTAINS "your search term" RETURN n, r, m`

EXAMPLES:

Question: "Tell me about Map Reduce"
Cypher: MATCH (n)-[r]-(m) WHERE toLower(n.id) CONTAINS "mapreduce" OR toLower(n.id) CONTAINS "map reduce" RETURN n, r, m

Question: "Who is Jeffrey Dean?"
Cypher: MATCH (n:Person)-[r]-(m) WHERE toLower(n.id) CONTAINS "jeffrey" RETURN n, r, m

Question: "What is HDFS used for?"
Cypher: MATCH (n)-[r]-(m) WHERE toLower(n.id) CONTAINS "hdfs" RETURN n, r, m

Question: "Details about the Mapreduce Library"
Cypher: MATCH (n)-[r]-(m) WHERE toLower(n.id) CONTAINS "mapreduce library" RETURN n, r, m

The question is:
{question}"""

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE
)

# --- QA TEMPLATE ---
QA_TEMPLATE = """You are Mimir, an expert assistant.
You have retrieved the following context from a Knowledge Graph to answer the user's question.

Context (Graph Data):
{context}

User Question:
{question}

INSTRUCTIONS FOR THE ANSWER:
1. **Analyze the Context:** Look at the relationships (e.g., USED_FOR, IMPLEMENTED_IN, AUTHOR_OF).
2. **Synthesize:** Don't just list facts. Combine them into coherent sentences.
3. **Be Comprehensive:** Use ALL the information provided in the context.
   - If the context says it is implemented in C++, say it.
   - If it says it is used for Google News, say it.
4. **Style:** Professional, technical, and detailed.

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_TEMPLATE
)

def main():
    # 1. SETUP ARGUMENTS
    parser = argparse.ArgumentParser(description="Mimir Chat: Ask questions to your Knowledge Graph.")
    parser.add_argument("--verbose", action="store_true", help="Show the generated Cypher queries")
    args = parser.parse_args()

    print(f"🤖 Initializing Mimir Chat Interface (Model: {MODEL_NAME})...")

    # 2. CONNECT TO NEO4J
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        # Refresh schema creates a map of your nodes/rels so the LLM knows what exists
        graph.refresh_schema()
        # IMPRIMIR EL ESQUEMA PARA QUE TÚ LO VEAS EN CONSOLA
        # print("\n--- SCHEMA DETECTED ---")
        # print(graph.schema)
        # print("-----------------------\n")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 3. SETUP THE BRAIN (LLM)
    llm = ChatOllama(model=MODEL_NAME, temperature=0, base_url="http://localhost:11434")

    # 4. CREATE THE CHAIN (The "Magic" Link)
    # GraphCypherQAChain does the following:
    # Question -> LLM translates to Cypher -> Neo4j executes -> Result -> LLM translates to Answer
    try:
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=args.verbose, # If True, it prints the thinking process
            allow_dangerous_requests=True, # Necessary for executing generated queries
            cypher_prompt=CYPHER_PROMPT,
            qa_prompt=QA_PROMPT
        )
    except Exception as e:
        print(f"❌ Error creating chain: {e}")
        return

    print("✅ Mimir is ready! (Type 'exit' or 'quit' to stop)")
    print("--------------------------------------------------")

    # 5. CHAT LOOP
    while True:
        try:
            user_input = input("\n🧑 You: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break

            # Run the chain
            response = chain.invoke({"query": user_input})

            # Identify the output (LangChain sometimes returns a dict or a string)
            answer = response.get("result", response)

            print(f"🤖 Mimir: {answer}")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("   (Tip: Sometimes the LLM generates invalid Cypher. Try rephrasing the question.)")

if __name__ == "__main__":
    main()