FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY backend backend
COPY frontend frontend
ENV PORT=8000
CMD ["sh","-c","uvicorn backend.main:app --host 0.0.0.0 --port $PORT"]
