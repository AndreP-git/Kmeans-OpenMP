#include <stdio.h>
#include <stdlib.h>
#include <string.h>     
#include <sys/types.h> 
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>
#include "kmeans.h"

int _debug;

static void print_help(char *argv0, float threshold)
{
    char *help =
        "Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -n num_clusters: number of clusters (K must > 1)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -p nproc       : number of threads (default system allocated)\n"
        "       -a             : perform ATOMIC version (default LOCAL)\n"
        "       -o             : output timing results (default no)\n"
        "       -d             : enable debug mode\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}


int main(int argc, char **argv)
{
    // Definition
    int opt, i, j, nthreads, atomic, time;
    extern char *optarg;
    char *filename;
    int n_clusters, n_coords, n_objs;
    float threshold;
    double  timing, IO_time, clustering_time;
    
    // Main structures
    float **objects; 
    float **clusters;
    int *membership; 

    // Initialization
    _debug            = 0;
    nthreads          = 0;
    n_clusters        = 0;
    threshold         = 0.001; // this changes the behaviour of the algorithm
    n_clusters        = 0;
    time              = 0;
    atomic            = 0;
    filename          = NULL;

    // Checking command line args
    while ( (opt=getopt(argc,argv,"p:i:n:t:ado"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'n': n_clusters = atoi(optarg);
                      break;
            case 'p': nthreads = atoi(optarg);
                      break;
            case 'a': atomic = 1;
                      break;
            case 'o': time = 1;
                      break;
            case 'd': _debug = 1;
                      break;
            case '?': print_help(argv[0], threshold);
                      break;
            default: print_help(argv[0], threshold);
                      break;
        }
    } //while

    // Invoking print_help if needed
    if (filename == 0 || n_clusters <= 1)
        print_help(argv[0], threshold);

    // If user specifies the no. threads in command line, else max threads allocated by run-time system
    if (nthreads > 0)
        omp_set_num_threads(nthreads);

    if (time)
        IO_time = omp_get_wtime();

    // Read data points from dataset
    objects = file_read(filename, &n_objs, &n_coords);
    if (objects == NULL)
        exit(EXIT_FAILURE);

    if (time) {
        timing = omp_get_wtime();
        IO_time = timing - IO_time;
        clustering_time = timing;
    }      

    // START
    membership = (int*) malloc(n_objs * sizeof(int));

    // Clustering process
    clusters = omp_kmeans(atomic, objects, n_coords, n_objs, n_clusters, threshold, membership);

    // STOP
    if (time) {
        timing = omp_get_wtime();
        clustering_time = timing - clustering_time;
    }         

    // Writing results
    file_write(filename, n_clusters, n_objs, n_coords, clusters, membership);

    // Releasing memory
    free(objects[0]);
    free(objects);  
    free(membership);
    free(clusters[0]);
    free(clusters);

    // Printing some stats
    if (time) {
        IO_time += omp_get_wtime() - timing;

        printf("\n**** Report: Kmeans using OpenMP ****\n");
        
        if (atomic)
            printf("Version: ATOMIC\n");
        else
            printf("Version: LOCAL\n");

        printf("\n----Data statistics----\n");
        printf("Input file:     %s\n", filename);
        printf("n_objs       = %d\n", n_objs);
        printf("n_coords     = %d\n", n_coords);

        printf("\n----Param statistics----\n");
        printf("Number of threads = %d\n", omp_get_max_threads());
        printf("n_clusters   = %d\n", n_clusters);
        printf("threshold     = %.4f\n", threshold);

        printf("\n----Time statistics----\n");
        printf("I/O time           = %10.4f sec\n", IO_time);
        printf("Computation timing = %10.4f sec\n", clustering_time);
    }

    return(0);
}