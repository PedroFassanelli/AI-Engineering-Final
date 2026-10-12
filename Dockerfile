# Una sola imagen para la API y el worker: cambia solo el comando.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # El codificador de tiktoken (para medir los fragmentos en tokens) se baja
    # en el build: el worker no depende de internet para eso al arrancar.
    TIKTOKEN_CACHE_DIR=/srv/.tiktoken

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY app ./app

# Usuario sin privilegios. /data es el volumen de la base de negocio (SQLite).
RUN useradd --create-home --uid 1000 nimbus \
    && mkdir -p /data && chown nimbus /data \
    && chmod -R a+rX /srv/.tiktoken
USER nimbus

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
