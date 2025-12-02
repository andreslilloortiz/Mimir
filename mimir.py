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

import streamlit as st
import os
import tempfile
from modules import database, ingestor, rag_engine
import config

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Mimir - GraphRAG",
    page_icon="🧙🏻‍♂️​",
    layout="wide"
)

def main():
    # --- SIDEBAR: CONFIGURATION & INGESTION ---
    with st.sidebar:
        st.header("Mimir Control Panel")

        # 1. Connection Check
        try:
            graph = database.get_graph_db()
            st.success("Neo4j Connected")
        except Exception as e:
            st.error(f"Connection Failed: {e}")
            st.stop()

        st.caption(f"Model: {config.MODEL_NAME}")

        st.markdown("---")
        st.markdown("### 📂 Ingest Documents")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        clear_db = st.checkbox("Clear Database before ingestion", value=False)

        if st.button("Start Ingestion"):
            if uploaded_file is not None:
                with st.spinner("Processing document... (Check terminal for logs)"):
                    try:
                        # Save uploaded file to temp
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name

                        # Clear DB if requested
                        if clear_db:
                            database.clear_database(graph)
                            st.toast("Database cleared!", icon="🧹")

                        # Run Ingestion Module
                        stats = ingestor.process_pdf(tmp_path, graph)

                        st.success(f"Ingestion Complete in {stats['duration']:.2f}s!")
                        st.markdown(f"**Pages:** {stats['pages']} | **Entities:** {stats['entities']}")

                        # Cleanup
                        os.remove(tmp_path)

                    except Exception as e:
                        st.error(f"Ingestion Error: {e}")
            else:
                st.warning("Please upload a file first.")

        st.markdown("---")
        st.markdown("### 🔍 Visualize Graph")
        st.caption("Inspect the generated nodes and relationships directly in Neo4j.")
        st.link_button("Open Neo4j Browser", "http://localhost:7474")

    # --- MAIN PAGE: CHAT INTERFACE ---
    st.title("🧙🏻‍♂️​ Chat with Mimir")
    st.caption("Graph Retrieval-Augmented Generation System")

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if prompt := st.chat_input("Ask a question about the graph..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                # Refresh Schema to ensure new data is visible
                graph.refresh_schema()

                # Get the RAG Chain
                chain = rag_engine.get_qa_chain(graph, verbose=True)

                # Execute Chain
                with st.spinner("Thinking..."):
                    response = chain.invoke({"query": prompt})
                    result = response.get("result", response)

                message_placeholder.markdown(result)

                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": result})

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()