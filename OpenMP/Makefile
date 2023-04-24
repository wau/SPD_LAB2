CC = gcc-12
CFLAGS = -Wall -Wextra -std=c99 -pedantic -fopenmp
LDFLAGS = -lm

SRC = dtw_omp.c

all: dtw_omp
.PHONY: all

run: tests
.PHONY: run

FILES=tests/10k.txt tests/20k.txt tests/30k.txt tests/40k.txt tests/50k.txt

clean:
	rm -f dtw_omp *.o

tests: dtw_omp
	for file in $(FILES); do \
    	echo "Running test with $$file"; \
    	./dtw_omp $$file $$file; \
		echo ""; \
	done

dtw_omp: $(SRC) Makefile
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)
