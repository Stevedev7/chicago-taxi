FROM python:3.8-slim
WORKDIR /dir
RUN pip install --upgrade google-cloud-bigquery
COPY requirements.txt /dir/
RUN pip install -r requirements.txt
COPY ./ /dir/
ENV GOOGLE_APPLICATION_CREDENTIALS=/dir/credentials.json