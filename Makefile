.DEFAULT_GOAL := help

# ── Docker ───────────────────────────────────────────────────────────────────

.PHONY: build up down logs bench stress info

build: ## Build the Docker image
	docker compose build

up: build ## Build + start benchmark (detached)
	docker compose up gpu-bench -d

down: ## Stop all services
	docker compose down

logs: ## Follow container logs
	docker compose logs -f

bench: build ## Run benchmark suite
	docker compose run --rm gpu-bench

stress: build ## Run full stress-test suite
	docker compose run --rm gpu-stress

info: build ## Print GPU info + telemetry snapshot
	docker compose run --rm gpu-info

# ── Local development ────────────────────────────────────────────────────────

.PHONY: install test test-stress test-info lint

install: ## Install locally via uv
	uv venv && uv pip install -e ".[dev]"

test: ## Run benchmark locally
	python -m gpu_test bench

test-stress: ## Run stress suite locally
	python -m gpu_test stress

test-info: ## Print GPU info locally
	python -m gpu_test info

lint: ## Syntax check all modules
	python3 -m py_compile gpu_test/__init__.py
	python3 -m py_compile gpu_test/__main__.py
	python3 -m py_compile gpu_test/utils.py
	python3 -m py_compile gpu_test/monitor.py
	python3 -m py_compile gpu_test/benchmark.py
	python3 -m py_compile gpu_test/stress.py
	@echo "✓ All modules OK"

# ── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
