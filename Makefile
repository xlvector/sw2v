CC = g++

sw2v : main.o ps-lite/build/libps.a
	$(CC) -o sw2v main.o ps-lite/build/libps.a -lboost_system -lstdc++ -L./ps-lite/deps/lib/ -lprotobuf-lite -lzmq -std=c++11 -DLOCAL -fopenmp

main.o : src/main.cc src/sw2v.cc include/sw2v.h ps-lite/build/libps.a
	$(CC) -c src/main.cc src/sw2v.cc -I./include -I./ps-lite/include -DLOCAL -std=c++11 -fopenmp

#sw2v.o : src/sw2v.cc include/sw2v.h ps-lite/build/libps.a
#	$(CC) -c -std=c++11 src/sw2v.cc -I./include -I./ps-lite/include -DLOCAL -fopenmp

clean: 
	rm -f *.o
	rm -f sw2v
