from python:3.7.8-stretch
RUN mkdir /app
COPY requirements.txt /app
WORKDIR /app
RUN apt-get update
RUN apt-get -y install libassimp-dev libgl1-mesa-dev mesa-utils libgl1-mesa-glx
RUN pip install -r requirements.txt
COPY . /app