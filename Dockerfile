FROM python:3.11-slim

WORKDIR /ai
COPY version_requirements.txt /ai/version_requirements.txt
RUN pip install --no-cache-dir -r /ai/version_requirements.txt

# 런타임에 바인드 마운트할 거라면 COPY 생략 가능

COPY worker.py /ai/worker.py
COPY ./models/multi_hand_gesture_classifier.tflite /ai/models/multi_hand_gesture_classifier.tflite

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "./worker.py"]