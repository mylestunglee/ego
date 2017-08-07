CC = g++
CFLAGS = -c -std=c++14 -Wall -Wextra -g
INC = -isystem eigen3 -isystem libgp/include -Igsl-2.4/gsl
LIBS = -Llibgp -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu/ -lgp -lpthread -lgsl -lgslcblas

SOURCES = tgp.cpp gp.cpp animation.cpp functions.cpp transferrer.cpp csv.cpp evaluator.cpp surrogate.cpp ego.cpp ihs.cpp main.cpp
HEADERS = tgp.hpp gp.hpp animation.hpp functions.hpp transferrer.hpp csv.hpp evaluator.hpp surrogate.hpp ego.hpp ihs.hpp main.hpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = ego

all: $(SOURCES) $(EXECUTABLE) $(HEADERS)

$(EXECUTABLE) : $(OBJECTS)
	$(CC) $(OBJECTS) $(INC) $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(INC) $< -o $@

clean:
	rm -rf *.o
	rm -f test $(EXECUTABLE) *.log


