CC = g++
CFLAGS = -Werror -Iinc/ -O0 -g
LDFLAGS = -g
SRCS = ${wildcard src/*.cpp}
OUT = main

DEPS= ${wildcard *.hpp}
SRCS= ${wildcard *.cpp}

all: default

default:
	${CC} ${CFLAGS} ${LDFLAGS} ${SRCS} ${DEPS} -o ${OUT} `pkg-config opencv4 --cflags --libs` -lpthread -mfpu=neon

clean:
	rm -f ${OUT}