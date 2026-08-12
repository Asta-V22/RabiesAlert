FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir gunicorn -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

COPY app.py db.py ./
COPY templates ./templates
COPY best_model.pth ./

ENV PORT=7860
EXPOSE 7860

CMD gunicorn -b 0.0.0.0:${PORT} -w 1 --threads 4 --timeout 120 app:app
