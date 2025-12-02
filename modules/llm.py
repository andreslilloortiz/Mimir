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

import requests
import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
import config

def check_and_pull_model(model_name):
    """
    Checks if the model exists locally in Ollama.
    If not, it triggers a download (pull) via the API.
    """
    base_url = config.OLLAMA_BASE_URL

    # 1. Get list of local models
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            # Extract model names (e.g., "llama3.2:latest")
            local_models = [m['name'] for m in models_data['models']]

            # Normalize names for comparison (Ollama adds :latest automatically)
            model_check = model_name if ":" in model_name else f"{model_name}:latest"

            # Check if model exists (checking for partial match to be safe)
            if any(model_check in m for m in local_models):
                return True # Model exists, no need to download

    except Exception as e:
        print(f"⚠️ Warning: Could not check local models: {e}")
        # We proceed to try creating the LLM anyway, letting LangChain handle errors
        return True

    # 2. If model is missing, trigger Pull
    print(f"⬇️ Model '{model_name}' not found locally. Starting download...")

    try:
        payload = {"name": model_name}
        # stream=True is important to prevent timeouts on large downloads
        with requests.post(f"{base_url}/api/pull", json=payload, stream=True) as r:
            r.raise_for_status()
            # Consume the stream to ensure download completes before proceeding
            for line in r.iter_lines():
                if line:
                    # Optional: Parse JSON line to show progress in logs
                    pass
        print(f"✅ Model '{model_name}' downloaded successfully!")
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to pull model {model_name}: {e}")

def get_llm(model_name=None, temperature=0):
    """
    Returns a configured ChatOllama instance.
    Automatically downloads the model if it is missing.
    """
    selected_model = model_name if model_name else config.DEFAULT_MODEL

    # --- AUTO-DOWNLOAD CHECK ---
    # This will block execution on the first run until the model is downloaded
    check_and_pull_model(selected_model)
    # ---------------------------

    return ChatOllama(
        model=selected_model,
        temperature=temperature,
        base_url=config.OLLAMA_BASE_URL
    )

def get_embeddings():
    """Returns the configured Embedding model."""
    # We ensure the embedding model is available (usually handled by Docker, but safe to check)
    check_and_pull_model(config.EMBEDDING_MODEL)

    return OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )