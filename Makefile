.PHONY: install test lint train api streamlit

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest -q

train:
	python src/models/train.py --data-path data/raw/train.csv --output models/xgb.joblib

api:
	uvicorn src.api.main:app --reload

streamlit:
	streamlit run src/app/streamlit_app.py