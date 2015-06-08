CC = g++
CFLAGS = -c -std=c++11 -Wall -Werror -Wmissing-prototypes
#INC = -I/usr/local/Cellar/eigen/3.2.4/include/eigen3/ -I../../libs/libgp/include -I../../libs/libgp/src -I../../libs/dlib-18.15 
INC = -I. -Ilibgp/include -Ilibgp/src 
LIBS =  -L../../libs/libgp/ -lgp

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
