# docker compose
docker compose up -d
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d # nvidia gpu
docker logs -f mimir-init # logs from llama3.2 download

# ingest
python3 ingest.py docs/document.pdf --clear

# python venv
python3 -m venv mimir-venv
source mimir-venv/bin/activate
pip install langchain langchain-community langchain-experimental langchain_ollama langchain_neo4j neo4j ollama pypdf tiktoken

# nvidia gpu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker