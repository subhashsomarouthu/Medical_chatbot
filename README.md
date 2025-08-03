# Memory-Optimized RAG Chatbot

A production-ready, memory-efficient Retrieval-Augmented Generation (RAG) system for biomedical question answering. This project leverages state-of-the-art LLMs, advanced semantic search, and privacy-aware filtering, and is fully Dockerized for easy deployment on cloud or local environments.

🚀 Quick Start
Prerequisites
Docker installed
At least 8GB RAM 
(Optional) NVIDIA GPU with CUDA support for faster inference
(Optional) Hugging Face account for gated models

Docker Deployment

Build the Docker image:

docker build -t rag-chatbot .


Run the container:

docker run -p 5000:5000 rag-chatbot


With Hugging Face token (for enhanced models):

docker run -p 5000:5000 -e HUGGINGFACE_HUB_TOKEN=your_token_here rag-chatbot


Check health status:

curl http://localhost:5000/health

📋 API Endpoints

Health Check:
GET /health
Returns system status and model loading information.

Predict (Main Endpoint):
POST /predict
Content-Type: application/json

{
  "query": "What is diabetes?"
}


Response Example:

{
  "response": "Generated medical response",
  "confidence": 0.85,
  "query": "What is diabetes?",
  "timestamp": "2024-01-15T10:30:00",
  "context_check": "passed",
  "model_used": "TinyLlama",
  "rewritten_query": "diabetes mellitus blood glucose insulin",
  "passages_found": 3
}


Model Information:
GET /info
Returns detailed model specifications and system information.

🏗️ Architecture
Embedder: sentence-transformers/all-MiniLM-L6-v2
Query Rewriter: google/gemma-2-2b-it (4-bit quantized)
Answer Generator: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Vector Store: FAISS index with 27,975 medical passages
Fallback Models: DistilGPT2, T5-small

Memory Optimizations:

4-bit quantization (BitsAndBytes)
Lazy model loading
GPU memory management (if available)
Efficient passage retrieval
🔒 Privacy & Security
Named Entity Recognition (NER):
Automatically detects and blocks questions seeking personal or sensitive information.
Out-of-Context Filtering:
Ensures only medical/health-related queries are answered.
Answer Truncation:
Returns only the most relevant, concise response for each query.

📊 Performance
Response Time: ~25 seconds (average)
Memory Usage: ~3-4GB RAM
GPU Usage: Optional, improves speed
Concurrent Users: Limited by available memory
Model Size: ~2GB compressed

Evaluation Metrics:

ROUGE-L (0.23)
BERT-F1 (85%)
MRR (75%)
MAP (73%)

📁 Project Structure
RAG_SYSTEM_CHECKPOINT/
├── app.py                          # Flask API server
├── chatbot.py                      # Main RAG logic
├── load_system.py                  # System loader
├── Dockerfile                      # Docker configuration
├── requirements.txt                # Python dependencies
├── model-info.json                 # API specifications
├── system_config.json              # System configuration
└── rag_system_checkpoint/          # Pre-trained models & data
    ├── passages_list.pkl
    ├── passage_ids.pkl
    ├── bioasq_faiss_index.index
    └── system_config.json

🔧 Configuration

Environment Variables:

HUGGINGFACE_HUB_TOKEN: Hugging Face token for gated models
FLASK_ENV: Set to 'development' for debug mode
PORT: Custom port (default: 5000)

Model Configuration:
Edit system_config.json to modify:

Model parameters (temperature, max_tokens)
Medical keyword filtering
Retrieval settings
🧪 Testing

Local Testing Example:

import requests

# Test health
response = requests.get('http://localhost:5000/health')
print(response.json())

# Test prediction
response = requests.post('http://localhost:5000/predict', 
                        json={'query': 'What is diabetes?'})
print(response.json())


Docker Testing:

docker exec <container_id> curl http://localhost:5000/health
docker logs <container_id>

🐳 Docker Commands
Build with custom tag:
docker build -t my-rag-system:v1.0 .

Run with custom configuration:
docker run -p 5000:5000 \
  -e HUGGINGFACE_HUB_TOKEN=your_token \
  -e FLASK_ENV=production \
  -v $(pwd)/logs:/app/logs \
  rag-chatbot

Save Docker image:
docker save rag-chatbot > rag-chatbot.tar

Load Docker image:
docker load < rag-chatbot.tar

🚨 Troubleshooting
Out of Memory Error:
Increase Docker memory allocation:
docker run -m 8g -p 5000:5000 rag-chatbot
Model Loading Timeout:
Check logs for progress:
docker logs -f <container_id>
CUDA Errors (GPU):
Run in CPU-only mode:
docker run -e CUDA_VISIBLE_DEVICES="" -p 5000:5000 rag-chatbot
Health Check Failures:
Wait 2-3 minutes for model loading, check logs, and verify memory allocation.


📝 Development

Adding New Models:

Update chatbot.py model loading section
Modify system_config.json parameters
Update model-info.json specifications
Rebuild Docker image

Custom Data:

Replace files in rag_system_checkpoint/
Update passage counts in config files
Rebuild FAISS index if needed



📣 Support

For issues or questions:

Check Docker logs: docker logs <container_id>
Verify system requirements (RAM, Docker version)
Test API endpoints manually with curl
Review this README for troubleshooting steps

Built with ❤️ by Subhash.

