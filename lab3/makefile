CC = g++
CFLAGS = -Werror
LDFLAGS = -g
SRCS = ${wildcard src/*.cpp}
OUT = build/main
BUILD = build

all: default

default:
	${CC} ${CFLAGS} ${LDFLAGS} ${SRCS} -o ${OUT} `pkg-config opencv4 --cflags --libs`

clean:
	rm build/main