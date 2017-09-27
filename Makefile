CC = g++
CFLAGS = -c -std=c++14 -Wall -Wextra -g
INC = -isystem eigen3 -isystem libgp/include -Igsl-2.4/gsl
LIBS = -Llibgp/build -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu/ -lgp -lpthread -lgsl -lgslcblas

SOURCES = compare.cpp tgp.cpp gp.cpp animation.cpp functions.cpp transferrer.cpp csv.cpp evaluator.cpp surrogate.cpp ego.cpp ihs.cpp
HEADERS = compare.hpp tgp.hpp gp.hpp animation.hpp functions.hpp transferrer.hpp csv.hpp evaluator.hpp surrogate.hpp ego.hpp ihs.hpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = ego

all: $(SOURCES) $(EXECUTABLE) $(HEADERS)

$(EXECUTABLE): $(OBJECTS) main.o
	$(CC) $(OBJECTS) main.o $(INC) $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(INC) $< -o $@

clean:
	rm -rf *.o
	rm -f test $(EXECUTABLE) *.log *.csv

test: $(OBJECTS) test.o
	$(CC) $(OBJECTS) test.o $(INC) $(LIBS) -o test
	./test
