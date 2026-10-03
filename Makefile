.PHONY: install cache train run streamlit benchmark test clean

install:
	pip install -r requirements.txt

cache:
	python main.py --build_cache

train:
	python -m src.train

run:
	python main.py --source 0

streamlit:
	streamlit run app.py

benchmark:
	python main.py --benchmark

test:
	python -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
