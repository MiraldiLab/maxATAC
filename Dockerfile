# Dockerfile, Image, Container
FROM python:3.9

WORKDIR /maxATAC

ADD maxatac .

COPY py3.9_requirements.txt py3.9_requirements.txt
RUN pip3 install -r py3.9_requirements.txt

CMD ["maxatac"]
