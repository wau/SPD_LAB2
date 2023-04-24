#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
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

int imin(int x, int y, int z){
    if (x <= y && x <= z) return x;
    else if (y <= x && y <= z) return y;
    else return z;
}

int dtw(int a[], int size_a, int b[], int size_b, int number_tasks, int rank, int num_threads){

    int from = rank-1;
    int to = rank+1;

    int msg = -1;

    MPI_Status Stat;

    int start_b = (size_b/number_tasks)*rank;
    int lastrank = (rank == number_tasks-1);

    int chunks = size_b/number_tasks;

    if (lastrank) chunks += size_b % number_tasks;
    int cols = 1 + chunks;
    int rows = 1 + size_a;

    //allocate dtw matrixes
    int **dtw_current = (int **)malloc(rows * sizeof(int *));
    int **dtw_previous = (int **)malloc(rows * sizeof(int *));
    for(int i=0; i<rows; i++) {
        dtw_current[i] = (int *)malloc(cols * sizeof(int));
        dtw_previous[i] = (int *)malloc(cols * sizeof(int));
    }

    // Initialize done matrix with infinity
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            dtw_previous[i][j] = INT_MAX;
        }
    }

    if(rank == 0) {
        dtw_previous[0][0] = 0;
        //calculate the cost matrix
        for(int i = 1; i < rows; i++){
            dtw_current[i][0] = INT_MAX;
            #pragma omp parallel for num_threads(num_threads) 
            for(int j = 1; j < cols; j++) {
                int cost = abs(a[i-1] - b[start_b + j-1]);
                dtw_current[i][j] = cost + imin(dtw_previous[i-1][j-1],dtw_previous[i-1][j],dtw_current[i][j-1]);
            }
            MPI_Send(&dtw_current[i][cols-1], 1, MPI_INT, to, 1, MPI_COMM_WORLD);

            int **temp = dtw_current;
            dtw_current = dtw_previous;
            dtw_previous = temp;
        }
    }
    else {
        for(int i = 1; i < rows; i++){
            MPI_Recv(&msg, 1, MPI_INT, from, 1, MPI_COMM_WORLD, &Stat);

            dtw_current[i][0] = msg;
            #pragma omp parallel for num_threads(num_threads)
            for(int j = 1; j < cols; j++) {
                int cost = abs(a[i-1] - b[start_b + j-1]);
                dtw_current[i][j] = cost + imin(dtw_previous[i-1][j-1],dtw_previous[i-1][j],dtw_current[i][j-1]);
            }

            if (!lastrank) MPI_Send(&dtw_current[i][cols-1], 1, MPI_INT, to, 1, MPI_COMM_WORLD);
            //swap
            int **temp = dtw_current;
            dtw_current = dtw_previous;
            dtw_previous = temp;
        }
    }

    int result = dtw_previous[rows-1][cols-1];

    // Free memory
    for(int i=0; i<rows; i++) {
        free(dtw_current[i]);
        free(dtw_previous[i]);
    }
    free(dtw_current);
    free(dtw_previous);

    if(lastrank) return result;

    MPI_Finalize();
    exit(0);
}

int main(int argc,char **argv){

    int *a = (int*)malloc(MAX_SIZE * sizeof(int));
    int *b = (int*)malloc(MAX_SIZE * sizeof(int));

    int size_a = read_arr_from_file(argv[1], a);
    int size_b = read_arr_from_file(argv[2], b);

    //swap arrays
    if(size_a > size_b){
        int tmp1 = size_a;
        size_a = size_b;
        size_b = tmp1;

        int *tmp = a;
        a = b;
        b = tmp;
    }
    MPI_Init(&argc,&argv);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //number of current task
    MPI_Comm_size(MPI_COMM_WORLD, &size); //number of tasks/jobs

    if( size < 2){
        if(rank == 0)
          printf("Error: you should use at least 2 tasks");
        exit(1);
    }

    int num_threads = atoi(argv[3]);

    double start = MPI_Wtime();
    int _dtw = dtw(a , size_a, b, size_b, size, rank, num_threads);
    double end = MPI_Wtime();
    double time_spent = (double)(end - start);
    printf("DTW distance = %d\n", _dtw);
    printf("Working time: %lf\n", time_spent);

    MPI_Finalize();
}