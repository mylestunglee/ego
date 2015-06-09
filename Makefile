CC = g++
CFLAGS = -c -std=c++0x -Wall #-Werror #-Wmissing-prototypes
INC = -Ieigen3 -Ilibgp/include -Ilibgp/src 
LIBS = -Llibgp -lgp -lpthread

SOURCES = surrogate.cpp functions.cpp bench.cpp optimise.cpp ego.cpp ihs.cpp libsvm-3.20/svm.cpp
HEADERS = surrogate.h optimise.h ego.h functions.h ihs.hpp libsvm-3.20/svm.h
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = test

all: $(SOURCES) $(EXECUTABLE) $(HEADERS)

$(EXECUTABLE) : $(OBJECTS)
	$(CC) $(OBJECTS) $(INC) $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(INC) $< -o $@

clean:
	rm -rf *.o
	rm -f test
