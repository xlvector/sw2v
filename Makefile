CC = g++

sw2v : main.o sw2v.o
	$(CC) -o sw2v main.o sw2v.o -lboost_system -lstdc++ -fopenmp

main.o : src/main.cc src/sw2v.cc include/sw2v.h
	$(CC) -c src/main.cc src/sw2v.cc -I./include -fopenmp

sw2v.o : src/sw2v.cc include/sw2v.h
	$(CC) -c src/sw2v.cc -I./include -fopenmp

clean: 
	rm -f *.o
	rm -f sw2v
