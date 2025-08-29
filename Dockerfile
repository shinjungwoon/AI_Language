FROM python:3.11-slim

WORKDIR /ai
COPY version_requirements.txt /ai/
RUN pip install --no-cache-dir -r /ai/version_requirements.txt

# 런타임에 바인드 마운트할 거라면 COPY 생략 가능
COPY worker.py /ai/worker.py
COPY . /ai

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "./worker.py"]