CC = g++
OUT = DisplayImage
LDFLAGS = -g
CFLAGS = -Werror

SRCS = ${wildcard *.cpp}

all: default

default:
	${CC} ${CFLAGS} ${LDFLAGS} ${SRCS} -o ${OUT} `pkg-config opencv4 --cflags --libs`

clean:
	rm -f ${OUT}
