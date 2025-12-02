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

# --- NEO4J CONFIGURATION ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password123"

# --- OLLAMA MODEL CONFIGURATION ---
DEFAULT_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
AVAILABLE_MODELS = [
    "llama3.2",      # 3B - Very fast, low VRAM (Best for Ingestion)
    "phi3:mini",     # 3.8B - Smart & Efficient (Microsoft)
    "mistral",       # 7B - Balanced performance
    "llama3.1:8b",   # 8B - Smartest, pushes VRAM limit (Best for Chat)
    "qwen2.5:7b"     # 7B - Great for coding/logic
]