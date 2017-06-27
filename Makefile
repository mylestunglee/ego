CC = g++
CFLAGS = -c -std=c++0x -Wall
INC = -isystem eigen3 -Ilibgp/include -Ilibgp/src -I/usr/include/python2.6 -I/usr/include/python2.7
LIBS = -Llibgp -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu/ -lgp -lpthread -lpython2.7

SOURCES = surrogate.cpp functions.cpp optimise.cpp ego.cpp ihs.cpp libsvm-3.20/svm.cpp main.cpp
HEADERS = surrogate.h optimise.h ego.h functions.h ihs.hpp libsvm-3.20/svm.h
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = pq_optimisation

all: $(SOURCES) $(EXECUTABLE) $(HEADERS)

$(EXECUTABLE) : $(OBJECTS)
	$(CC) $(OBJECTS) $(INC) $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(INC) $< -o $@

clean:
	rm -rf *.o
	rm -f test
	rm $(EXECUTABLE)

old_test: main.o $(OBJECTS)
	$(CC) main.o surrogate.o functions.o optimise.o ego.o ihs.o libsvm-3.20/svm.o $(INC) $(LIBS) -o old_test


