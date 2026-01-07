.PHONY: setup run clean

setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Setup complete."

run:
	@echo "Starting OCR API..."
	@echo "Setting environment variables..."
	@set PYTHONIOENCODING=utf-8
	@set FLAGS_use_mkldnn=0
	python ocr_api.py

clean:
	@echo "Cleaning up..."
	@if exist "__pycache__" rd /s /q "__pycache__"
	@del /q *.pyc
	@echo "Clean complete."
