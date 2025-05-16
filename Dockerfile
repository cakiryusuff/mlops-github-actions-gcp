FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -e .

EXPOSE 5000

ENV FLASK_APP=application.py

# Container başladığında önce eğit, sonra API'yi başlat
CMD ["sh", "-c", "python pipeline/training_pipeline.py && python application.py"]