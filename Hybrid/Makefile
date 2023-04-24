CC = mpicc
CFLAGS = -fopenmp
LDFLAGS = -lm -lmpi -fopenmp

SRC = dtw_hybrid.c

all: dtw_hybrid
.PHONY: all

run: tests
.PHONY: run

FILES=tests/5k.txt tests/10k.txt tests/20k.txt tests/40k.txt tests/80k.txt tests/100k.txt tests/120k.txt tests/140k.txt

NUM_PROCESSES=4

clean:
	rm -f dtw_hybrid *.o

tests: dtw_hybrid
	for file in $(FILES); do \
    	echo "Running test with $$file"; \
    	mpirun --use-hwthread-cpus -np $(NUM_PROCESSES) ./dtw_hybrid $$file $$file; \
		echo ""; \
	done

dtw_hybrid: $(SRC) Makefile
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)
