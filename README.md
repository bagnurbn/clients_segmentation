python -m venv .venv

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit jupyter


Python: 3.11.9