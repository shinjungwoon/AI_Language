FROM python:3.11-slim

WORKDIR /app
COPY AI_Language/version_requirements.txt /app/
RUN pip install --no-cache-dir -r /app/version_requirements.txt

# 런타임에 바인드 마운트할 거라면 COPY 생략 가능
COPY ai_worker/worker.py /app/ai_worker/worker.py
COPY ai_worker/AI_Language /app

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "ai_worker/worker.py"]