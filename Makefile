.DEFAULT_GOAL := help
.PHONY: default lint clean clean-pyc clean-test
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

clean: clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
