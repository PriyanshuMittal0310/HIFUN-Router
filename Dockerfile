FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Spark/PySpark runtime prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy current project state (including strict curated artifacts)
COPY . /app

# Default command: print quality gate summary artifact location.
CMD ["bash", "-lc", "echo 'Image built with HIFUN Router project state at /app'; ls -1 training_data/dataset_quality_report_strict_curated.json experiments/results/relevance_eval_strict.json experiments/results/ablation_strict.json"]
