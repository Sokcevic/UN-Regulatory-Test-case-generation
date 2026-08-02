FROM python:3.12-slim

WORKDIR /app

# System deps: beautifulsoup4/pymupdf wheels are prebuilt, but keep a C toolchain
# around in case any dependency needs to build from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY regulatory_testgen/requirements.txt /app/regulatory_testgen/requirements.txt
COPY chat_ui/requirements.txt /app/chat_ui/requirements.txt

RUN pip install --no-cache-dir \
    -r regulatory_testgen/requirements.txt \
    -r chat_ui/requirements.txt

COPY regulatory_testgen /app/regulatory_testgen
COPY chat_ui /app/chat_ui
COPY data /app/data

EXPOSE 8501

CMD ["streamlit", "run", "chat_ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
