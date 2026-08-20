.PHONY: help install test run retrain clean lint

help:
	@echo "SignalForge Commands"
	@echo "===================="
	@echo "make install      - Install dependencies from requirements.txt"
	@echo "make test         - Run all tests (27 passing tests)"
	@echo "make test-cov     - Run tests with coverage report"
	@echo "make run          - Start the app (loads saved model if available)"
	@echo "make retrain      - Force model retraining on startup"
	@echo "make lint         - Check syntax with py_compile"
	@echo "make clean        - Remove cache files and compiled artifacts"

install:
	pip install -r requirements.txt

test:
	pytest test_arnn_model.py test_app.py -v

test-cov:
	pytest test_arnn_model.py test_app.py -v --cov=. --cov-report=html

run:
	python app.py

retrain:
	RETRAIN=1 python app.py

lint:
	python -m py_compile app.py arnn_model.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete"
