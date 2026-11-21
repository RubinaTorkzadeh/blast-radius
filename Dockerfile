FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
COPY coverage_map.json.gz /app/coverage_map.json.gz
COPY ast_map.json.gz /app/ast_map.json.gz
COPY dialect_map.json.gz /app/dialect_map.json.gz
COPY selector.py /app/selector.py
RUN chmod 644 /app/coverage_map.json.gz /app/ast_map.json.gz /app/dialect_map.json.gz /app/selector.py
CMD ["python", "/app/selector.py"]
