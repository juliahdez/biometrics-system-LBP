FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_LOG_LEVEL=ERROR

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Usuario sin privilegios; datos/ y secretos/ se montan desde fuera
RUN useradd --create-home usuario \
 && mkdir -p datos secretos \
 && chown -R usuario:usuario datos secretos
USER usuario

VOLUME ["/app/datos", "/app/secretos"]
ENTRYPOINT ["python", "biometria.py"]
CMD ["--help"]
