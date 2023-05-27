FROM python:3

COPY ./model /app/model
COPY ./test_inference.py /app
WORKDIR /app

RUN pip install numpy tensorflow Pillow

ENTRYPOINT ["python", "./test_inference.py"]
