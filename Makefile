CC = g++

sw2v : main.o sw2v.o ps-lite/build/libps.a
	$(CC) -o sw2v main.o sw2v.o ps-lite/build/libps.a -lboost_system -lstdc++ -L./ps-lite/deps/lib/ -lprotobuf-lite -lzmq -std=c++11

main.o : src/main.cc src/sw2v.cc include/sw2v.h ps-lite/build/libps.a
	$(CC) -c -std=c++11 src/main.cc src/sw2v.cc -I./include -I./ps-lite/include

sw2v.o : src/sw2v.cc include/sw2v.h ps-lite/build/libps.a
	$(CC) -c src/sw2v.cc -I./include -I./ps-lite/include -std=c++11

clean: 
	rm -f *.o
	rm -f sw2v
