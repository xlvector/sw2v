FROM index.alauda.cn/featured/phusion-baseimage

RUN apt-get update && apt-get install -y git

RUN apt-get install -y build-essential wget libboost-dev libboost-system-dev

RUN git clone --recursive https://github.com/xlvector/sw2v /tmp/sw2v

RUN cd /tmp/sw2v/ps-lite && make

RUN echo "begin build 2"

RUN cd /tmp/sw2v && make -f Makefile.ps

RUN mkdir -p /mnt/dcos

ENTRYPOINT ["/tmp/sw2v/sw2v"]

