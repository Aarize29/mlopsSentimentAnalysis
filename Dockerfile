FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN python sentiment_analysis.py


CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
