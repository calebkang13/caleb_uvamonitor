FROM mysql:5.7.24
WORKDIR /home/monitor_db
RUN apt-get update
RUN apt-get -y install python3 python3-pip
RUN pip3 install sklearn scipy tqdm numpy
COPY . .
EXPOSE 3306
