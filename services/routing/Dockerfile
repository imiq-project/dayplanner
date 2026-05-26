FROM python:3.13
WORKDIR /app
RUN pip install flask
COPY *.json .
COPY main.py .
CMD [ "python", "main.py" ]
