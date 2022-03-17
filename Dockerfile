FROM python:3.8.13-slim-bullseye

RUN apt-get update && apt-get install -y git python3-dev gcc procps htop \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -U pip wheel setuptools
RUN pip install --upgrade -r requirements.txt

COPY app app/

RUN python app/server.py

EXPOSE 55563 55564

CMD ["python", "app/server.py", "serve"]
