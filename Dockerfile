FROM python:3.8.13-slim-bullseye

# All these env variable declarations are necessary to receive variables from Docker Compose during the build phase.
ARG FLASK_PORT
ENV FLASK_PORT $FLASK_PORT
ARG WEB_PORT
ENV WEB_PORT $WEB_PORT
ARG MODEL_REPOSITORY_URI
ENV MODEL_REPOSITORY_URI $MODEL_REPOSITORY_URI
ARG MODEL_PATH
ENV MODEL_PATH $MODEL_PATH
ARG MODEL_FILENAME
ENV MODEL_FILENAME $MODEL_FILENAME

RUN apt-get update && apt-get install -y git python3-dev gcc procps htop \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -U pip wheel setuptools
RUN pip install --upgrade -r requirements.txt

COPY app app/

RUN python app/server.py

EXPOSE $FLASK_PORT $WEB_PORT

# After the build phase is concluded, environment variables are taken from the docker compose env_file key (== variables.env) and used in the CMD line
CMD [ "bash", "-c", "python app/server.py --cmd serve --model-url $MODEL_REPOSITORY_URI/$MODEL_PATH/$MODEL_FILENAME --model-name $MODEL_FILENAME --flask-port $FLASK_PORT --web-port $WEB_PORT"]
