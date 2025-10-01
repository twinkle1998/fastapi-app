# at2weather

Custom Python package for the Advanced ML AT2 assignment.

Includes:
- Data fetching (Open-Meteo API)
- Feature engineering (classification + regression)
- Model training utilities (XGBoost, CatBoost)
- Metrics, persistence, and plotting helpers


API Deployment

The FastAPI app is in a separate repo:
twinkle1998/fastapi-app

Endpoints:
	•	/ → Overview + endpoints.
	•	/health/ → Health check.
	•	/predict/rain/?date=YYYY-MM-DD → Predict rain in 7 days.
	•	/predict/precipitation/fall/?date=YYYY-MM-DD → Predict 3-day precipitation sum.

Deployed on Render → public endpoint available after deployment.

⸻

 Troubleshooting

Git Issues
	•	CRLF/LF warnings: harmless, Windows auto-converts line endings.
	•	“remote contains work that you do not have locally”: run

git pull origin main --allow-unrelated-histories
git push -u origin main --force



TestPyPI Publishing
	•	Error “File already exists” → bump version in pyproject.toml.

[tool.poetry]
name = "at2weather"
version = "0.1.2"


	•	Error “Requires-Python >=3.10,<3.11” → adjust Python version in Poetry config to match your interpreter.

Render Deployment
	•	Build command:

pip install -r requirements.txt


	•	Start command:

uvicorn app.main:app --host 0.0.0.0 --port 10000


	•	Add at2weather inside repo so Render doesn’t miss the package.

⸻
