#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <omp.h>

#define THREADS 4
#define MAX_SIZE 40005


void print_arr(const int *a, int n) {
  if (n > 0)
  {
    printf("%d", a[0]);
    for (int i = 1; i < n; i++)  // i = 1
      printf(" %d", a[i]);
  }
  printf("\n");
}

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

int DTWP(int a[], int size_a, int b[], int size_b) {
    int** DTW = (int**)malloc((size_a+1) * sizeof(int*));
    for (int i = 0; i <= size_a; i++) {
        DTW[i] = (int*)malloc((size_b+1) * sizeof(int));
    }

    int i, j, cost;

    
    for (i = 1; i <= size_a; i++)
        DTW[i][0] = INT_MAX;
    for (i = 1; i <= size_b; i++)
        DTW[0][i] = INT_MAX;
    DTW[0][0] = 0;


    //int nthreads, tid;
    #pragma omp parallel shared(DTW) num_threads(THREADS)
    {
        
        #pragma omp for ordered
        for (i = 1; i <= size_a; i++) {
            #pragma omp ordered
            for (j = 1; j <= size_b; j++) {
                cost = abs(a[i-1] - b[j-1]);
                DTW[i][j] = cost + fmin(DTW[i-1][j], fmin(DTW[i][j-1], DTW[i-1][j-1]));
            }
        }
    }
    
    return DTW[size_a][size_b];
}






void files_input(char *file1, char *file2) {

//    p_array = (int *)malloc(sizeof(int)*50);      // allocate 50 ints

    //int arr1[MAX_SIZE];
    //int arr2[MAX_SIZE];

    //printf("%s\n", file1);
    //printf("%s\n", file2);


    //int *arr1 = (int*)malloc(MAX_SIZE * sizeof(int));
    //int *arr2 = (int*)malloc(MAX_SIZE * sizeof(int));
    int arr1[MAX_SIZE];
    int arr2[MAX_SIZE];


    int size1 = read_arr_from_file(file1, arr1);
    int size2 = read_arr_from_file(file2, arr2);

    

    double start, end;


    start = omp_get_wtime();

    

    int distance = DTWP(arr1, size1, arr2, size2);

    end = omp_get_wtime();


    printf("DTW distance = %d\n", distance);
    printf("Work took %f seconds\n", end - start);

}


int main( int argc , char * argv []) {
    //tests();
    files_input(argv[1], argv[2]);
    (void)argc; // avoid warning error
    
    return 0;
}