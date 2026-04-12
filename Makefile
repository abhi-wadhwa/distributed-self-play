.PHONY: install dev test lint format clean docker-build docker-up docker-down demo

# Install the package
install:
	pip install -e .

# Install with dev dependencies
dev:
	pip install -e ".[dev]"

# Run all tests
test:
	pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# Lint with ruff
lint:
	ruff check src/ tests/

# Auto-format with ruff
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Build Docker image
docker-build:
	docker build -t distributed-self-play .

# Start all services
docker-up:
	docker compose up -d

# Scale actors
docker-scale:
	docker compose up -d --scale actor=$(N)

# Stop all services
docker-down:
	docker compose down

# View logs
docker-logs:
	docker compose logs -f

# Run quick demo (no Redis needed)
demo:
	python -m src.cli demo --game connect4 --simulations 25 --num-games 2

# Run demo with Othello
demo-othello:
	python -m src.cli demo --game othello --simulations 15 --num-games 1
