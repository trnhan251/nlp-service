FROM python:3.8.8
ADD . /python-flask
WORKDIR /python-flask
RUN pip install -r requirements.txt