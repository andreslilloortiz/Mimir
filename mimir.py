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
from modules import database, ingestor, rag_engine, llm
import config

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Mimir - GraphRAG",
    page_icon="🧙🏻‍♂️",
    layout="wide"
)

def main():
    # --- SIDEBAR: CONFIGURATION & INGESTION ---
    with st.sidebar:
        st.header("Mimir Control Panel")

        # Connection Check
        try:
            graph = database.get_graph_db()
            st.success("Neo4j Connected")
        except Exception as e:
            st.error(f"Connection Failed: {e}")
            st.stop()

        # Model selector
        st.markdown("### 🧠 LLM Brain")
        selected_model = st.selectbox(
            "Select Model",
            config.AVAILABLE_MODELS,
            index=0,
            help="Choose the LLM used for extraction and chat."
        )

        is_installed = llm.is_model_available(selected_model)
        if is_installed:
            st.success(f"**{selected_model}** is ready to use.")
        else:
            st.warning(f"**{selected_model}** not found. It will be downloaded automatically on first use (this may take time).")

        # Ingest
        st.markdown("---")
        st.markdown("### 📂 Ingest Documents")
        uploaded_file = st.file_uploader(
            "Upload File",
            type=["pdf", "docx", "txt", "md"],
            help="Supported formats: PDF, Word, Text, Markdown"
        )

        clear_db = st.checkbox("Clear Database before ingestion", value=False)

        if st.button("Start Ingestion"):
            if uploaded_file is not None:
                # Indicate potentially long download time in the spinner
                with st.spinner(f"Preparing {selected_model} (downloading if missing) & Processing..."):
                    try:
                        # Detect extension to save temp file correctly
                        file_ext = os.path.splitext(uploaded_file.name)[1]

                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name

                        if clear_db:
                            database.clear_database(graph)
                            st.toast("Database cleared!", icon="🧹")

                        # Pass the selected model to the Ingestor
                        stats = ingestor.process_file(tmp_path, graph, model_name=selected_model)

                        st.success(f"Ingestion Complete in {stats['duration']:.2f}s!")
                        st.markdown(f"**Chunks/Pages:** {stats['pages']} | **Entities:** {stats['entities']}")
                        os.remove(tmp_path)

                    except Exception as e:
                        st.error(f"Ingestion Error: {e}")
            else:
                st.warning("Please upload a file first.")

        # Graph Neo4j
        st.markdown("---")
        st.markdown("### 🔍 Visualize Graph")
        st.caption("Inspect the generated nodes and relationships directly in Neo4j.")
        st.link_button("Open Neo4j Browser", "http://localhost:7474")

    # --- MAIN PAGE: CHAT INTERFACE ---
    st.title("🧙🏻‍♂️ Chat with Mimir")
    st.caption(f"Graph Retrieval-Augmented Generation System | Powered by {selected_model}")

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

            with st.spinner(f"{selected_model} is thinking (downloading if missing)..."):
                try:
                    graph.refresh_schema()

                    # Initialize Hybrid Engine
                    rag_system = rag_engine.get_qa_chain(graph, model_name=selected_model)

                    # Execute Hybrid Query
                    result = rag_system.query(prompt)

                    message_placeholder.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()