CC = mpicc
LDFLAGS = -lm -lmpi

SRC = dtw_mpi.c

all: dtw_mpi
.PHONY: all

run: tests
.PHONY: run

FILES=tests/10k.txt tests/20k.txt tests/30k.txt tests/40k.txt tests/50k.txt


NUM_PROCESSES=4

clean:
	rm -f dtw_mpi *.o

tests: dtw_mpi
	for file in $(FILES); do \
    	echo "Running test with $$file"; \
    	mpirun --use-hwthread-cpus -np $(NUM_PROCESSES) ./dtw_mpi $$file $$file; \
		echo ""; \
	done

single: dtw_mpi
    echo "Running test with 10k.txt; \
    	mpirun --use-hwthread-cpus -np $(NUM_PROCESSES) ./dtw_mpi tests/10k.txt tests/10k.txt; \
		echo ""; \
	done

dtw_mpi: $(SRC) Makefile
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)
