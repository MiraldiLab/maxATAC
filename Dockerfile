# Dockerfile, Image, Container
FROM python:3.9

#RUN cd maxATAC && pip3 install . -c py3.9_requirements.txt

WORKDIR /maxATAC

#ADD maxatac .

#COPY py3.9_requirements.txt py3.9_requirements.txt
COPY . .
RUN pip3 install -r py3.9_requirements.txt && pip3 install -e .

#COPY . .

CMD ["maxatac"]
