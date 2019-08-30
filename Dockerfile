FROM python:3.7

ADD ./src/ /app/

RUN pip install -r /app/requirements.txt --no-cache-dir --compile