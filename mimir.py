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
from streamlit_option_menu import option_menu
import os
import tempfile
from modules import database, ingestor, rag_engine, llm, analytics
import config

# --- VISUAL CONFIGURATION ---
st.set_page_config(
    page_title="Mimir",
    page_icon="🧙🏻‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # 1. SIDEBAR
    with st.sidebar:
        st.header("🧙🏻‍♂️ Mimir")
        st.caption("Experimental Hybrid GraphRAG Pipeline with Local LLMs.")

        st.markdown("---")

        # NAVIGATION MENU
        with st.container():
            view = option_menu(
                menu_title=None,
                options=["Chat", "Ingest", "Analytics"],
                icons=["chat-quote", "file-earmark-arrow-up", "graph-up-arrow"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": "#fafafa", "font-size": "18px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#333333"},
                    "nav-link-selected": {"background-color": "#724BFF"},
                }
            )

        st.markdown("---")

        # Model Selector
        st.caption("AI BRAIN")
        selected_model = st.selectbox(
            "Select Model",
            config.AVAILABLE_MODELS,
            index=0,
            label_visibility="collapsed"
        )

        if llm.is_model_available(selected_model):
            st.caption(f"🟢 **{selected_model}** ready")
        else:
            st.caption(f"🟠 **{selected_model}** will download on use")

        st.markdown("---")

        # Database Status & Link
        try:
            database.get_graph_db()
            st.caption("🟢 Neo4j Online · [Open Browser](http://localhost:7474)")
        except:
            st.error("🔴 Neo4j Disconnected")

    # 2. VIEW: CHAT INTERFACE
    if view == "Chat":
        st.subheader(f"Chat with {selected_model}")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input Area
        if prompt := st.chat_input("Ask about your knowledge base..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner(f"{selected_model} is thinking..."):
                    try:
                        graph = database.get_graph_db()
                        graph.refresh_schema()

                        # Initialize Hybrid RAG
                        rag = rag_engine.get_qa_chain(graph, model_name=selected_model)
                        response_data = rag.query(prompt)

                        answer_text = response_data["answer"]
                        sources = response_data["sources"]

                        # 1. Render the main answer
                        placeholder.markdown(answer_text)

                        # 2. Render sources
                        if sources:
                            with st.expander("📚 Reference Sources"):
                                for i, doc in enumerate(sources):
                                    # Clean up filename
                                    source_name = os.path.basename(doc.get('source', 'Unknown'))
                                    page = doc.get('page', 'N/A')
                                    # Preview content (first 250 chars)
                                    content_preview = doc.get('content', '')[:250].replace('\n', ' ')

                                    st.markdown(f"**{i+1}. {source_name}** (Page {page})")
                                    st.caption(f"_{content_preview}..._")
                                    if i < len(sources) - 1:
                                        st.divider()

                        # 3. Save only text to history to keep context clean
                        st.session_state.messages.append({"role": "assistant", "content": answer_text})

                    except Exception as e:
                        placeholder.error(f"Error: {e}")

# 3. VIEW: DOCUMENT MANAGEMENT
    elif view == "Ingest":
        st.subheader("Knowledge Ingestion")

        with st.expander("⚙️ Advanced Settings"):
            clear_db = st.toggle("Clear existing database before ingestion")

        tab_file, tab_url = st.tabs(["📄 File Upload", "🌐 Web URL"])

        # --- TAB 1: ARCHIVOS ---
        with tab_file:
            uploaded_file = st.file_uploader(
                "Upload documents",
                type=["pdf", "docx", "txt", "md"],
                label_visibility="collapsed"
            )

            if uploaded_file and st.button("Process Document", type="primary", use_container_width=True):
                graph = database.get_graph_db()

                with st.status("Running File Ingestion Pipeline...", expanded=True) as status:
                    try:
                        st.write("📂 Saving temporary file...")
                        ext = os.path.splitext(uploaded_file.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        if clear_db:
                            st.write("🧹 Wiping Neo4j database...")
                            database.clear_database(graph)

                        st.write(f"🧠 Extracting Graph & Vectors using {selected_model}...")
                        stats = ingestor.process_file(tmp_path, graph, model_name=selected_model, original_filename=uploaded_file.name)

                        os.remove(tmp_path)
                        status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)

                        st.divider()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Time", f"{stats['duration']:.2f}s")
                        col2.metric("Chunks", stats['pages'])
                        col3.metric("Entities", stats['entities'])

                    except Exception as e:
                        status.update(label="❌ Ingestion Failed", state="error")
                        st.error(f"Error: {e}")

        # --- TAB 2: URLS ---
        with tab_url:
            target_url = st.text_input("Enter URL", placeholder="https://en.wikipedia.org/wiki/Graph_database")

            if target_url and st.button("Process URL", type="primary", use_container_width=True):
                graph = database.get_graph_db()

                with st.status("Running Web Ingestion Pipeline...", expanded=True) as status:
                    try:
                        st.write(f"🌐 Fetching content from {target_url}...")

                        if clear_db:
                            st.write("🧹 Wiping Neo4j database...")
                            database.clear_database(graph)

                        st.write(f"🧠 Extracting Graph & Vectors using {selected_model}...")

                        stats = ingestor.process_url(target_url, graph, model_name=selected_model)

                        status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)

                        st.divider()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Time", f"{stats['duration']:.2f}s")
                        col2.metric("Chunks", stats['pages'])
                        col3.metric("Entities", stats['entities'])

                    except Exception as e:
                        status.update(label="❌ Ingestion Failed", state="error")
                        st.error(f"Error: {e}")

    # 4. VIEW: ANALYTICS (NEW)
    elif view == "Analytics":
        st.subheader("Graph Analytics")
        st.caption("Discover hidden patterns using Neo4j Graph Data Science algorithms.")

        # Connect to DB
        try:
            graph = database.get_graph_db()

            # A. General Stats
            stats = analytics.get_stats(graph)
            col1, col2 = st.columns(2)
            col1.metric("Total Nodes", stats['nodes'])
            col2.metric("Total Relationships", stats['edges'])

            st.divider()

            # B. Trigger Button
            if st.button("Run Deep Analysis", type="primary"):
                with st.spinner("Calculating PageRank and Communities..."):

                    # 1. PageRank
                    st.markdown("### 🏆 Top Influential Concepts (PageRank)")
                    df_pr = analytics.run_pagerank(graph)

                    if not df_pr.empty:
                        # Bar Chart
                        st.bar_chart(df_pr.set_index("Entity"), color="#724BFF")
                    else:
                        st.warning("Not enough data. Ingest more documents first.")

                    st.divider()

                    # 2. Communities
                    st.markdown("### 🧩 Detected Topic Clusters (Louvain)")
                    df_comm = analytics.run_community_detection(graph)

                    if not df_comm.empty:
                        st.dataframe(
                            df_comm,
                            column_config={
                                "Community": "ID",
                                "Members": st.column_config.NumberColumn("Size"),
                                "Examples": "Sample Entities"
                            },
                            width='stretch',
                            hide_index=True
                        )
                    else:
                        st.info("No clear communities found yet.")

        except Exception as e:
            st.error(f"Analytics Error: {e}")

if __name__ == "__main__":
    main()