# KumoRFM Makefile

.PHONY: help install install-dev test lint format clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install      Install KumoRFM"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code with black"
	@echo "  clean        Clean build artifacts"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=kumorfm --cov-report=html

lint:
	flake8 kumorfm/ tests/
	mypy kumorfm/

format:
	black kumorfm/ tests/ examples/
	isort kumorfm/ tests/ examples/

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t kumorfm:latest .

docker-run:
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/checkpoints:/app/checkpoints \
		kumorfm:latest

# Development helpers
run-demo:
	python examples/demo.py

jupyter:
	jupyter notebook --notebook-dir=notebooks/

docs:
	cd docs && make html

# RelGT integration
install-relgt:
	git clone https://github.com/snap-stanford/relgt.git models/relgt/relgt
	pip install -r models/relgt/relgt/requirements.txt