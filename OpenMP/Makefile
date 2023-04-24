CC = gcc-12
CFLAGS = -Wall -Wextra -std=c99 -pedantic -fopenmp
LDFLAGS = -lm

SRC = dtw_omp.c

all: dtw_omp
.PHONY: all

run: tests
.PHONY: run

FILES=tests/5k.txt tests/10k.txt tests/20k.txt tests/40k.txt tests/80k.txt tests/100k.txt tests/120k.txt tests/140k.txt

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
