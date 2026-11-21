FROM python:3.13-slim

WORKDIR /app
COPY coverage_map.json.gz /app/coverage_map.json.gz
COPY ast_map.json.gz /app/ast_map.json.gz
COPY dialect_map.json.gz /app/dialect_map.json.gz
COPY fixture_map.json.gz /app/fixture_map.json.gz
COPY selector.py /app/selector.py
RUN chmod 644 /app/coverage_map.json.gz /app/ast_map.json.gz /app/dialect_map.json.gz /app/fixture_map.json.gz /app/selector.py

CMD ["python3", "/app/selector.py"]
