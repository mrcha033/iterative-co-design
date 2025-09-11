FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && \
    pip install .

# Default command prints schema then exits; override in docker run
CMD ["python", "-m", "icd.cli.main", "run", "--print-schema", "-c", "configs/mock.json", "--out", "/tmp/ignore"]

