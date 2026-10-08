FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# lite requirements keep the image small; switch to requirements.txt for local sentence-transformers
COPY requirements-lite.txt .
RUN pip install -r requirements-lite.txt
COPY . .
EXPOSE 8501 8000
# APP=streamlit (default) or APP=api
ENV APP=streamlit
CMD if [ "$APP" = "api" ]; then uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}; \
    else streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0; fi
