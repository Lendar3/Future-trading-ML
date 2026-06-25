FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir -e .

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

