# Spark Chatbot 🚀

A specialized chatbot that answers Apache Spark questions using RAG (Retrieval Augmented Generation) with official Apache Spark documentation.

## Features

- **RAG-powered responses**: Uses official Apache Spark documentation for accurate answers
- **Real-time chat interface**: Interactive web UI for asking questions
- **LLM Integration**: Supports OpenAI, Anthropic, or local LLMs
- **Vector database**: Efficient document retrieval with Chroma/FAISS
- **REST API**: FastAPI backend for scalability
- **Docker support**: Easy deployment

## Tech Stack

- **Backend**: Python, FastAPI
- **Database**: Chroma (Vector DB)
- **LLM**: OpenAI API / Anthropic Claude / Local Models
- **Frontend**: HTML, CSS, JavaScript (React optional)
- **Deployment**: Docker, Docker Compose

## Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose (optional)
- OpenAI API key or alternative LLM credentials

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/C-sandeep-kumar/Spark-Chatbot.git
cd Spark-Chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r backend/requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Run the application**
```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`
Swagger docs: `http://localhost:8000/docs`

### Docker Setup

```bash
docker-compose up --build
```

## API Endpoints

### POST /chat
Ask a question about Apache Spark

**Request:**
```json
{
  "query": "How do I create a DataFrame in PySpark?",
  "chat_history": []
}
```

**Response:**
```json
{
  "answer": "To create a DataFrame in PySpark...",
  "sources": [{"title": "...", "url": "...", "relevance_score": 0.95}],
  "confidence": 0.95
}
```

### GET /health
Health check endpoint

## Configuration

Edit `backend/config.py` to customize:
- LLM provider (OpenAI, Anthropic, etc.)
- Vector database settings
- Document chunk size
- Number of documents to retrieve

## Project Structure

```
Spark-Chatbot/
├── backend/
│   ├── main.py                      # FastAPI application
│   ├── requirements.txt               # Python dependencies
│   ├── config.py                      # Configuration
│   ├── rag_engine.py                  # RAG logic
│   ├── spark_docs_loader.py           # Download & process Spark docs
│   ├── llm_provider.py                # LLM integration
│   └── schemas.py                     # Pydantic models
├── frontend/
│   ├── index.html                    # Chat UI
│   ├── style.css                     # Styling
│   └── script.js                     # Client logic
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

## Environment Variables

Create a `.env` file in the root directory:

```env
# LLM Configuration
LLM_PROVIDER=openai  # Options: openai, anthropic, local
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here

# Spark Docs
SPARK_DOCS_URL=https://spark.apache.org/docs/latest/

# Vector DB
VECTOR_DB_PATH=./data/vectordb

# App
DEBUG=True
PORT=8000
```

## Usage Examples

### Example 1: Basic Question
```
Q: "What is Apache Spark?"
A: Apache Spark is an open-source unified analytics engine...
```

### Example 2: Code Help
```
Q: "How do I join two DataFrames in PySpark?"
A: You can join DataFrames using the join() method...
```

### Example 3: Performance Tips
```
Q: "How can I optimize Spark performance?"
A: Here are some optimization techniques...
```

## Development

### Adding New Data Sources

Edit `backend/spark_docs_loader.py` to add custom Spark documentation or other resources.

### Customizing RAG Settings

Modify `backend/rag_engine.py`:
- Adjust chunk size for document splitting
- Change similarity threshold
- Modify number of retrieved documents

### Testing

```bash
cd backend
pytest tests/
```

## Deployment

### Deploy on Heroku
```bash
heroku create your-spark-chatbot
git push heroku main
```

### Deploy on AWS/GCP
Docker image is provided for easy containerization.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review Apache Spark official docs: https://spark.apache.org/docs/latest/

## Roadmap

- [ ] Support for Spark Scala documentation
- [ ] Multi-language support
- [ ] Advanced chat history management
- [ ] Rate limiting & authentication
- [ ] Analytics dashboard
- [ ] Mobile app

---

**Built with ❤️ for Apache Spark enthusiasts**