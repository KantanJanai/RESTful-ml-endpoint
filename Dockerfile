# Dockerfile to build a flask app

FROM python:3.8
WORKDIR /usr/app

copy . .

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
