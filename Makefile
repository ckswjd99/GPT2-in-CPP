CXX=g++
OBJDIR=./obj/
COMMON=-Iinclude/
BLASLIB=openblas
DEBUG=0
OPTS=
SOURCES=$(wildcard src/*.cpp)
OBJS=$(patsubst src/%.cpp, $(OBJDIR)%.o, $(SOURCES))
LDFLAGS=-l$(BLASLIB) -lm

APPLE=$(shell uname -a | grep -q "Darwin" && echo 1 || echo 0)
ifeq ($(APPLE), 1)
	COMMON+=-I/opt/homebrew/opt/openblas/include
	LDFLAGS+=-L/opt/homebrew/opt/openblas/lib
else
	COMMON+=-I/usr/include/openblas
	LDFLAGS+=-L/usr/lib/openblas
endif

ifeq ($(DEBUG), 1)
	CXX+=-g
	OPTS+=-DDEBUG
endif

DEPS = $(wildcard src/*.hpp) $(wildcard include/*.hpp) Makefile

make: $(OBJS)
	$(CXX) $(COMMON) $(OPTS) main.cpp $^ -o main.out $(LDFLAGS)

$(OBJS): $(OBJDIR)%.o: src/%.cpp $(DEPS) $(OBJDIR)
	$(CXX) $(COMMON) $(OPTS) -c $< -o $@ $(LDFLAGS)

$(OBJDIR):
	mkdir obj

clean:
	rm -rf $(OBJDIR) main.out
