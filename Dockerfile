FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860 PYTHONUNBUFFERED=1
RUN python -m app.startup_gate
CMD ["sh","-c","python -m app.startup_gate && uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
