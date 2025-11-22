Credit Card Clients Segmentation (K-Means + Streamlit)

├─ О проекте
Цель — построить end-to-end решение: от кластеризации клиентов (K-Means) до веб-интерфейса на Streamlit для классификации новых клиентов в реальном времени.
Алгоритм: K-Means на стандартизированных признаках.
Данные: Kaggle — Credit Card Dataset for Clustering (артикул CC GENERAL.csv).
Фронт: Streamlit с аккуратным вводом признаков (слайдеры 0..1 для частот, числовые поля для сумм).

├─ Структура
streamlit_app.py – фронт.
kmeans_model.pkl – модель.
scaler.pkl – скейлер.
model.ipynb – обучение модели.
requirements.txt – используемые библиотеки в проекте.

├─ Инструкция по пользованию:
Пройдите по ссылке: https://clientssegmentation-frxnfu8klaxdgaflghynkl.streamlit.app/
В форме заполните поля.
Нажмите Predict Cluster.
