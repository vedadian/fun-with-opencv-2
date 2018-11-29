CC = g++
CFLAGS = --std=c++11 -g -Wall
SRCS = fun.cpp
PROG = fun

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

all: $(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
