FROM python:3.10-slim-bookworm

COPY . /app
WORKDIR /app

RUN pip install poetry==1.5.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

EXPOSE 8000

CMD ["poetry", "run", "python", "main.py"]