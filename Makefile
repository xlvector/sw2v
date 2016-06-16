CC = clang

sw2v : main.o sw2v.o
	$(CC) -o sw2v main.o sw2v.o -lboost_system -lstdc++ -Wc++11-extensions

main.o : src/main.cc src/sw2v.cc include/sw2v.h
	$(CC) -c src/main.cc src/sw2v.cc -I./include

sw2v.o : src/sw2v.cc include/sw2v.h
	$(CC) -c src/sw2v.cc -I./include

clean: 
	rm -f *.o
	rm -f sw2v
