FROM python:3.10

WORKDIR /app

COPY api/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY temp_gradscii /app/temp_gradscii
COPY api/main.py /app/api/

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
