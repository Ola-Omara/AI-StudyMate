FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV VECTOR_STORE_PATH=/data/vector_store
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV FRONTEND_ORIGIN=http://localhost:8501
ENV APP_ENV=production

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
