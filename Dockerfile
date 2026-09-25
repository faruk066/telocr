# Railway / Render / herhangi bir konteyner ortamı için.
# Polling aynen çalışır; ek servis gerekmez.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# openpyxl/pandas için sistem bağımlılığı gerekmez; derleme araçsız kurulum için:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py ai_parser.py bot.py excel_maker.py ./

# Çalışma klasörleri (geçici fotoğraf + xlsx)
RUN mkdir -p /app/temp

# Render/Railway $PORT verir ama polling kullandığımız için dinlemiyoruz;
# sağlık kontrolü isteyen platformlar için bilgi amaçlı:
EXPOSE 8000

CMD ["python", "bot.py"]
