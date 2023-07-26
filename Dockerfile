FROM python:3.8.10
#ADD source/main.py

WORKDIR /

ADD source source
ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt
CMD ["python", "./source/main.py"]