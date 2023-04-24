CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -pedantic
LDFLAGS = -lm

SRC = dtw_seq.c

all: dtw_seq
.PHONY: all

run: tests
.PHONY: run

FILES=tests/5k.txt tests/10k.txt tests/20k.txt tests/40k.txt tests/80k.txt tests/100k.txt tests/120k.txt tests/140k.txt

clean:
	rm -f dtw_seq *.o

tests: dtw_seq
	for file in $(FILES); do \
    	echo "Running test with $$file"; \
    	./dtw_seq $$file $$file; \
		echo ""; \
	done

dtw_seq: $(SRC) Makefile
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)
