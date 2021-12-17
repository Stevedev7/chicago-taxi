FROM python:3.8-slim
WORKDIR /dir
COPY ./requirements.txt /dir/
RUN pip install -r requirements.txt
RUN pip install sklearn --upgrade
COPY ./ /dir/