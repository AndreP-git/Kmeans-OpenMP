#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "kmeans.h"


__inline static // warnings
float distance(int n_dim, float *coord1, float *coord2) // Squared Euclidean distance, other distances can be implemented
{
    int i;
    float res=0.0;

    // Iterating over dimensions
    for (i=0; i<n_dim; i++)
        res += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return(res);
}


__inline static // warnings
int find_nearest_cluster(int n_clusters, int n_coords, float *object, float **clusters)
{
    int index, i;
    float dist, min_dist;

    // Finding cluster with minimum distance to object
    index = 0;
    min_dist = distance(n_coords, object, clusters[0]);

    // Iterating over clusters
    for (i=1; i<n_clusters; i++) {

        dist = distance(n_coords, object, clusters[i]);
        
        // If current distance is smaller than min_dist, then update min_dist and belonging cluster
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }

    } // for

    return(index);

}

// main function
float** omp_kmeans(int atomic, float **objects, int n_coords, int n_objs, int n_clusters, float threshold, int *membership)
{
    // Defining structures
    int i, j, k, index, loop=0;
    int *newClusterSize; // Clusters dim

    float delta; // objects that change cluster
    float **clusters;
    float **newClusters; 
    double timing;
    int n_threads;

    // Threads local variables
    int **local_newClusterSize;
    float ***local_newClusters;    // [n_threads][n_clusters][n_coords] 

    // Retrieving maximum number of threads available
    n_threads = omp_get_max_threads();

    // Initializing clusters structure
    clusters = (float**) malloc(n_clusters * sizeof(float*));
    clusters[0] = (float*) malloc(n_clusters * n_coords * sizeof(float));
    for (i=1; i<n_clusters; i++)
        clusters[i] = clusters[i-1] + n_coords;

    // Choosing initial cluster centers (e.g. first K data points, other strategies are possible)
    for (i=0; i<n_clusters; i++)
        for (j=0; j<n_coords; j++)
            clusters[i][j] = objects[i][j];

    // Initializing membership structure with value -1
    for (i=0; i<n_objs; i++)
        membership[i] = -1;

    // Initializing newClusterSize and newClusters[0] with value 0
    newClusterSize = (int*) calloc(n_clusters, sizeof(int));
    newClusters = (float**) malloc(n_clusters * sizeof(float*));
    newClusters[0] = (float*) calloc(n_clusters * n_coords, sizeof(float));
    for (i=1; i<n_clusters; i++)
        newClusters[i] = newClusters[i-1] + n_coords;

    // LOCAL
    if (!atomic) {
        // Each thread calculates new centers using a private space, then thread 0 does an array reduction on them
        // This approach should be faster

        // Initializing local_newClusterSize
        local_newClusterSize = (int**) malloc(n_threads * sizeof(int*));
        local_newClusterSize[0] = (int*) calloc(n_threads*n_clusters, sizeof(int));
        for (i=1; i<n_threads; i++)
            local_newClusterSize[i] = local_newClusterSize[i-1] + n_clusters;

        // Initializing local_newClusters
        local_newClusters = (float***) malloc(n_threads * sizeof(float**));
        local_newClusters[0] =(float**) malloc(n_threads * n_clusters * sizeof(float*));
        for (i=1; i<n_threads; i++)
            local_newClusters[i] = local_newClusters[i-1] + n_clusters;

        // Iterating over threads
        for (i=0; i<n_threads; i++) {
            // Iterating over clusters
            for (j=0; j<n_clusters; j++)
                local_newClusters[i][j] = (float*) calloc(n_coords, sizeof(float));
        }

    } // !atomic

    if (_debug) timing = omp_get_wtime();

    do {
        delta = 0.0;
        // ATOMIC
        if (atomic) {

            #pragma omp parallel for \
                    private(i,j,index) \
                    firstprivate(n_objs,n_clusters,n_coords) \
                    shared(objects,clusters,membership,newClusters,newClusterSize) \
                    schedule(static) \
                    reduction(+:delta)

            // Iterating over objects        
            for (i=0; i<n_objs; i++) {

                // Finding index of nestest cluster center
                index = find_nearest_cluster(n_clusters, n_coords, objects[i], clusters);

                // If membership changes, increase delta by 1 
                if (membership[i] != index)
                    delta += 1.0;

                // Assigning membership to object i 
                membership[i] = index;

                // Updating newClusterSize and newCluster must be done one thread at a time //
                #pragma omp atomic                                                          //
                newClusterSize[index]++;                                                    //            
                                                                                            //    
                for (j=0; j<n_coords; j++)                                                  //                
                    #pragma omp atomic                                                      //
                    newClusters[index][j] += objects[i][j];                                 //                    
                // ------------------------------------------------------------------------ //

            } // for

        }
        else { // LOCAL

            #pragma omp parallel shared(objects,clusters,membership,local_newClusters,local_newClusterSize)
            {
                int tid = omp_get_thread_num();

                #pragma omp for \
                            private(i,j,index) \
                            firstprivate(n_objs,n_clusters,n_coords) \
                            schedule(static) \
                            reduction(+:delta)

                // Iterating over objects 
                for (i=0; i<n_objs; i++) {

                    // Finding index of nestest cluster center
                    index = find_nearest_cluster(n_clusters, n_coords, objects[i], clusters);

                    // If membership changes, increase delta by 1 
                    if (membership[i] != index) delta += 1.0;

                    // Assigning membership to object i 
                    membership[i] = index;

                    // Updating local_newClusterSize and local_newCluster (average will be performed later)
                    local_newClusterSize[tid][index]++;
                    for (j=0; j<n_coords; j++)
                        local_newClusters[tid][index][j] += objects[i][j];

                } // for

            } // #pragma omp parallel

            // Main thread performs the array reduction 
            for (i=0; i<n_clusters; i++) { // clusters
                for (j=0; j<n_threads; j++) { // threads

                    newClusterSize[i] += local_newClusterSize[j][i];
                    local_newClusterSize[j][i] = 0.0;

                    for (k=0; k<n_coords; k++) { // coordinates
                        newClusters[i][k] += local_newClusters[j][i][k];
                        local_newClusters[j][i][k] = 0.0;

                    }
                } // for threads
            } // for cluster

        } // else

        // Computing means and replacing old cluster centers with newClusters
        for (i=0; i<n_clusters; i++) { // clusters
            for (j=0; j<n_coords; j++) { // coordinates

                if (newClusterSize[i] > 1)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0;  
            }

            newClusterSize[i] = 0;
        } // for clusters

        // Computing delta to evaluate do-while condition
        delta = delta / n_objs;

    } while (delta > threshold && loop++ < 500); // added a utility termination condition

    if (_debug) {
        timing = omp_get_wtime() - timing;
        printf("nloops = %2d (T = %7.4f)",loop,timing);
    }

    // Releasing memory
    if (!atomic) {
        free(local_newClusterSize[0]);
        free(local_newClusterSize);

        for (i=0; i<n_threads; i++)
            for (j=0; j<n_clusters; j++)
                free(local_newClusters[i][j]);
        free(local_newClusters[0]);
        free(local_newClusters);
    }
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}