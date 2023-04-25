#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

#define MAX_SIZE 150000

int read_arr_from_file(char* filename, int* arr) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Could not open file %s\n", filename);
        return -1;
    }

    int size = 0;
    int num;
    while (fscanf(file, "%d", &num) == 1) {
        arr[size++] = num;
    }

    fclose(file);

    return size;
}

int imin(int x, int y, int z) {
    if (x <= y && x <= z) return x;
    else if (y <= x && y <= z) return y;
    else return z;
}

int dtw(int a[], int size_a, int b[], int size_b, int number_tasks, int rank) {
    int from = rank - 1;
    int to = rank + 1;

    int msg = -1;
    MPI_Status Stat;

    int start_b = (size_b / number_tasks) * rank;
    int lastrank = (rank == number_tasks - 1);

    int chunks = size_b / number_tasks;
    if (lastrank) chunks += size_b % number_tasks;

    int cols = 1 + chunks;
    int rows = 2;

    // Allocate DTW matrixes
    int *dtw_current = (int *) malloc(cols * sizeof(int));
    if (dtw_current == NULL) {
        printf("Error: Failed to allocate memory for dtw_current.\n");
        exit(EXIT_FAILURE);
    }
    int *dtw_previous = (int *) malloc(cols * sizeof(int));
    if (dtw_previous == NULL) {
        printf("Error: Failed to allocate memory for dtw_previous.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize done matrix with infinity
    for (int j = 0; j < cols; j++) {
        dtw_previous[j] = INT_MAX;
    }
    dtw_previous[0] = 0;

    if (rank == 0) {
        for (int i = 1; i <= size_a; i++) {
            dtw_current[0] = INT_MAX;
            int start_j = 1;
            if (i == 1) start_j = 0;

            for (int j = start_j; j < cols; j++) {
                int cost = abs(a[i - 1] - b[start_b + j - 1]);
                dtw_current[j] = cost + imin(dtw_previous[j - 1], dtw_previous[j], dtw_current[j - 1]);
            }

            MPI_Send(&dtw_current[cols - 1], 1, MPI_INT, to, 1, MPI_COMM_WORLD);

            // Swap
            int *temp = dtw_current;
            dtw_current = dtw_previous;
            dtw_previous = temp;
        }
    } else {
        for (int i = 1; i <= size_a; i++) {
            MPI_Recv(&msg, 1, MPI_INT, from, 1, MPI_COMM_WORLD, &Stat);

            dtw_current[0] = msg;
            int start_j = 1;
            if (i == 1) start_j = 0;
	    for (int j= start_j; j < cols; j++) {
		int cost = abs(a[i - 1] - b[start_b + j - 1]);
		dtw_current[j] = cost + imin(dtw_previous[j - 1], dtw_previous[j], dtw_current[j - 1]);
		}
	    if (!lastrank) MPI_Send(&dtw_current[cols - 1], 1, MPI_INT, to, 1, MPI_COMM_WORLD);

        // Swap
        int *temp = dtw_current;
        dtw_current = dtw_previous;
        dtw_previous = temp;
    }
}

int result = dtw_previous[cols - 1];

	// Free memory
	free(dtw_current);
	free(dtw_previous);

	if (lastrank) return result;

	MPI_Finalize();
	exit(0);
}
int main(int argc, char **argv) {
int *a = (int*)malloc(MAX_SIZE * sizeof(int));
int *b = (int*)malloc(MAX_SIZE * sizeof(int));
int size_a = read_arr_from_file(argv[1], a);
int size_b = read_arr_from_file(argv[2], b);

// Swap arrays if necessary
if (size_a > size_b) {
    int tmp1 = size_a;
    size_a = size_b;
    size_b = tmp1;

    int *tmp = a;
    a = b;
    b = tmp;
}

MPI_Init(&argc, &argv);
int rank, size;

MPI_Comm_rank(MPI_COMM_WORLD, &rank); //number of current task
MPI_Comm_size(MPI_COMM_WORLD, &size); //number of tasks/jobs

if (size < 2) {
    if (rank == 0) printf("Error: you should use at least 2 tasks");
    exit(1);
}

double start = MPI_Wtime();
int _dtw = dtw(a, size_a, b, size_b, size, rank);
double end = MPI_Wtime();
double time_spent = (double) (end - start);
printf("DTW distance = %d\n", _dtw);
printf("Working time: %lf\n", time_spent);

MPI_Finalize();
return 0;
}