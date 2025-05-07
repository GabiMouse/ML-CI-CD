FROM python:3.9-slim

# katalog roboczy w kontenerze
WORKDIR /app

# kopiowanie plików aplikacji do kontenera
COPY . /app

# zależności Pythona
RUN pip install --no-cache-dir -r requirements.txt

# port aplikacji
EXPOSE 5000

# uruchomienie aplikacji Flask
CMD ["python", "app.py"]
