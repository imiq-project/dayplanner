FROM python:3.13
WORKDIR /app
RUN pip install flask jsonschema
COPY main.py .
COPY templates/ templates/
CMD [ "python", "main.py" ]
