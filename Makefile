.DEFAULT_GOAL := help
.PHONY: default lint
SHELL := /bin/bash


default: help;

help:
	@echo "🔥 NOS - Nitrous Oxide for your AI infrastructure."
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  clean               Remove all build, test, coverage and Python artifacts"
	@echo "  lint                Format source code automatically"
	@echo ""

lint: ## Format source code automatically
	pre-commit run --all-files # Uses pyproject.toml
