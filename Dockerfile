FROM index.alauda.cn/featured/phusion-baseimage

RUN apt-get update && apt-get install -y git

RUN apt-get install -y build-essential

RUN git clone --recursive https://github.com/xlvector/sw2v

RUN apt-get install -y wget

RUN cd sw2v/ps-lite && make

RUN apt-get install -y libboost-dev libboost-system-dev

RUN cd sw2v && make -f Makefile.ps

ENTRYPOINT ["./sw2v/dcos.sh"]