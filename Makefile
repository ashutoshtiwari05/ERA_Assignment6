VENV_NAME := venv-a6
VENV_BIN := $(VENV_NAME)/bin
PYTHON := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip

.PHONY: setup clean train env-update clean-checkpoints activate check-model check-all view-results test test-model test-training

activate:
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV_BIN)/activate"
	@source $(VENV_BIN)/activate

setup:
	python -m venv $(VENV_NAME)
	@echo "Creating virtual environment and installing dependencies..."
	. $(VENV_BIN)/activate && $(PIP) install --upgrade pip && $(PIP) install -r requirements.txt
	@echo "\nSetup complete! Run 'make activate' to see activation instructions"

env-update:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt --upgrade
	@echo "\nEnvironment packages updated!"
	$(PYTHON) environment.py

clean: clean-checkpoints
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	@echo "Cleaned up Python cache files"

clean-checkpoints:
	rm -rf checkpoints/*
	@echo "Cleaned up checkpoints"

train:
	. $(VENV_BIN)/activate && python train.py

# New commands for model checks
check-model:
	@echo "Running model architecture checks..."
	@mkdir -p .github/scripts
	@$(PYTHON) .github/scripts/check_model.py
	@echo "\nCheck results:"
	@cat model_check_results.txt

check-params:
	@echo "Checking model parameters..."
	@$(PYTHON) -c "from model import Net; import torch; model = Net(); params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f'\nTotal trainable parameters: {params:,}')"
	@$(PYTHON) -c "from model import Net; import torch; model = Net(); params = sum(p.numel() for p in model.parameters() if p.requires_grad); exit(1 if params > 20000 else 0)" || echo "\n❌ Warning: Parameter count exceeds 20,000!"

check-all: check-model check-params
	@echo "\nAll checks completed!"

view-results:
	@if [ -f results.json ]; then \
		python -c "import json; \
			f = open('results.json', 'r'); \
			results = json.load(f); \
			f.close(); \
			print('\nBest Results:'); \
			print('-' * 50); \
			best = max(results, key=lambda x: x['test_accuracy']); \
			print(f\"Epoch: {best['epoch']}\"); \
			print(f\"Validation Accuracy: {best['val_accuracy']:.2f}%\"); \
			print(f\"Test Accuracy: {best['test_accuracy']:.2f}%\")"; \
	else \
		echo "No results file found"; \
	fi

test-model:
	@echo "Running model tests..."
	$(PYTHON) -m pytest tests/test_model.py -v

test-training:
	@echo "Running training tests..."
	$(PYTHON) -m pytest tests/test_training.py -v

test: test-model test-training
	@echo "\nAll tests completed!"