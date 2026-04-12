FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Default command (overridden by docker-compose)
CMD ["python", "-m", "src.cli", "--help"]
