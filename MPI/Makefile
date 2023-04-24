CC = mpicc
LDFLAGS = -lm -lmpi

SRC = dtw_mpi.c

all: dtw_mpi
.PHONY: all

run: tests
.PHONY: run

FILES=tests/5k.txt tests/10k.txt tests/20k.txt tests/40k.txt tests/80k.txt tests/100k.txt tests/120k.txt tests/140k.txt


NUM_PROCESSES=4

clean:
	rm -f dtw_mpi *.o

tests: dtw_mpi
	for file in $(FILES); do \
    	echo "Running test with $$file"; \
    	mpirun --use-hwthread-cpus -np $(NUM_PROCESSES) ./dtw_mpi $$file $$file; \
		echo ""; \
	done

dtw_mpi: $(SRC) Makefile
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)