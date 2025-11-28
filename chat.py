import argparse
from langchain_ollama import ChatOllama
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

# --- CONFIGURATION ---
# Connection details for the Dockerized Neo4j instance
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password123"
MODEL_NAME = "llama3.2"

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
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 3. SETUP THE BRAIN (LLM)
    llm = ChatOllama(model=MODEL_NAME, temperature=0, base_url="http://localhost:11434")

    # 4. CREATE THE CHAIN (The "Magic" Link)
    # GraphCypherQAChain does the following:
    # Question -> LLM translates to Cypher -> Neo4j executes -> Result -> LLM translates to Answer
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=args.verbose, # If True, it prints the thinking process
        allow_dangerous_requests=True # Necessary for executing generated queries
    )

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