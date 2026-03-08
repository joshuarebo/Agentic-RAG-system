FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/chroma_data

# Pre-download the embedding model so first request is fast
RUN python -c "import chromadb; c = chromadb.Client(); col = c.get_or_create_collection('warmup'); col.add(ids=['w'], documents=['warmup'])"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
