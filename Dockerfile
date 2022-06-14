FROM python:3.9-slim-buster AS builder

ARG PIPSERVER_URL 
ARG PIPSERVER_USERNAME 
ARG PIPSERVER_PASSWORD

WORKDIR /app

RUN python -m pip install --upgrade pip && \
    python -m pip install poetry && \
    poetry config repositories.omnia $PIPSERVER_URL && \
    poetry config http-basic.omnia $PIPSERVER_USERNAME $PIPSERVER_PASSWORD && \
    python -m venv /venv

COPY poetry.lock pyproject.toml ./

RUN . /venv/bin/activate && poetry install --no-dev --no-root

COPY . .

RUN . /venv/bin/activate && poetry build

FROM python:3.9-slim-buster

WORKDIR /app

COPY --from=builder /venv /venv
COPY --from=builder /app/dist .

RUN . /venv/bin/activate && pip install *.whl && \
    rm -f *.whl && \
    rm -f *.tar.gz