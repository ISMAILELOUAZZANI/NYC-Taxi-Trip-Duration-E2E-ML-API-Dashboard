```markdown
# NYC Taxi Trip Duration — End-to-End ML + API + Dashboard

Forecast taxi trip duration (seconds) from features like pickup/dropoff coordinates, datetime, passenger_count.

Project highlights:
- Feature engineering: Haversine distance, time features
- Modeling: XGBoost with Optuna hyperparameter tuning
- Explainability: SHAP ready
- Serving: FastAPI endpoint
- What-if Dashboard: Streamlit app
- Reproducibility: DVC + MLflow (hooks + examples)
- CI: GitHub Actions
- Containerized: Docker + docker-compose

Repository layout

nyc-taxi-duration/
├─ data/                # use DVC or keep out of git
│  ├─ raw/              # train.csv, test.csv
│  └─ processed/
├─ notebooks/
│  └─ 01_eda.ipynb
├─ src/
│  ├─ features/
│  │  ├─ geo.py
│  │  └─ build_features.py
│  ├─ models/
│  │  ├─ train.py
│  │  ├─ predict.py
│  │  └─ evaluate.py
│  ├─ api/
│  │  └─ main.py
│  └─ app/
│     └─ streamlit_app.py
├─ tests/
│  └─ test_features.py
├─ configs/
│  └─ config.yaml
├─ models/              # saved models (gitignored)
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ Makefile
├─ .pre-commit-config.yaml
├─ .github/workflows/ci.yaml
├─ .gitignore
└─ README.md

```

Quickstart (local)
1. Put Kaggle `train.csv` into `data/raw/` (or use Kaggle API + DVC).
2. Create and activate venv, then:
   - pip install -r requirements.txt
3. Run tests:
   - make test
4. Train a quick model (example)
   - python src/models/train.py --data-path data/raw/train.csv --output models/xgb.joblib
5. Serve API:
   - uvicorn src.api.main:app --reload
6. Run Streamlit dashboard:
   - streamlit run src/app/streamlit_app.py

See configs/config.yaml for default parameters.

License: MIT
```
