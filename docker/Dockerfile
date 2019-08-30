FROM python:3.7

ADD ./container_start.sh /start/
ADD ./requirements-small.txt /start/

RUN ["chmod", "+x", "/start/container_start.sh"]
RUN pip install -r /start/requirements-small.txt --no-cache-dir --compile