FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[app]"

# Optional NSE tick data is mounted at /data (see docker-compose.yml).
# With nothing mounted the dashboard starts on the built-in synthetic market.
ENV STRATBT_DATA_DIR=/data \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["stratbt", "app", "--port", "8501"]
