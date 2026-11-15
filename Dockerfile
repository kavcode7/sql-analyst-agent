FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install dependencies first so Docker can cache this layer between code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY web/ web/

# Unpack the demo database (made by scripts/make_demo_db.py) to the path the code expects.
COPY deploy/olist_demo.db.gz /tmp/olist_demo.db.gz
RUN mkdir -p data \
 && python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('/tmp/olist_demo.db.gz'), open('data/olist.db', 'wb'))" \
 && rm /tmp/olist_demo.db.gz

EXPOSE 8000
# Hosts like Render and Railway tell the app which port to use through $PORT.
CMD ["sh", "-c", "uvicorn api:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]
