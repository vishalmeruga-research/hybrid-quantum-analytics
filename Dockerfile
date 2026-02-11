FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -e ".[qiskit,pennylane]"

CMD ["qabda", "run", "--config", "examples/config_local.yaml"]
