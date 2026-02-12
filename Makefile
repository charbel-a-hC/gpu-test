.DEFAULT_GOAL := help

# Pass extra args via ARGS=  e.g.  make stress ARGS="--tests throughput kv_cache --duration 120"
ARGS ?=

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

bench: build ## Run benchmark suite (ARGS= for extra flags)
	docker compose run --rm gpu-bench gpu-test bench $(ARGS)

stress: build ## Run stress suite (ARGS="--tests all --duration 60")
	docker compose run --rm gpu-stress gpu-test stress $(ARGS)

info: build ## Print GPU info + telemetry snapshot
	docker compose run --rm gpu-info

# ── Local development ────────────────────────────────────────────────────────

.PHONY: install test test-stress test-info lint

install: ## Install locally via uv
	uv venv && uv pip install -e ".[dev]"

test: ## Run benchmark locally (ARGS= for extra flags)
	python -m gpu_test bench $(ARGS)

test-stress: ## Run stress suite locally (ARGS= for extra flags)
	python -m gpu_test stress $(ARGS)

test-info: ## Print GPU info locally
	python -m gpu_test info

lint: ## Syntax check all modules
	python3 -m py_compile gpu_test/__init__.py
	python3 -m py_compile gpu_test/__main__.py
	python3 -m py_compile gpu_test/utils.py
	python3 -m py_compile gpu_test/monitor.py
	python3 -m py_compile gpu_test/benchmark.py
	python3 -m py_compile gpu_test/stress.py
	python3 -m py_compile gpu_test/report.py
	@echo "✓ All modules OK"

# ── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
