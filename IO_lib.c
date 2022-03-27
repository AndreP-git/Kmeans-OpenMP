#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "kmeans.h"

#define MAX_LEN 128

// Utility function to read a file in ASCII format
float** file_read(char *filename, int  *n_objs, int  *n_coords)
{  
    // Structures definition
    float **objects;
    int i, j, len;
    FILE *fp;
    char *line, *ret;
    int line_len;

    // Opening file + check
    if ((fp = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        return NULL;
    }

    // Finding the number of objects 
    line_len = MAX_LEN;
    line = (char*) malloc(line_len);

    (*n_objs) = 0;

    // Iterating over lines of the input file
    while (fgets(line, line_len, fp) != NULL) {

        // Checking each line to find the correct max length
        while (strlen(line) == line_len-1) {

            len = strlen(line);
            fseek(fp, -len, SEEK_CUR);

            // Increase line_len 
            line_len += MAX_LEN;
            line = (char*) realloc(line, line_len);

            ret = fgets(line, line_len, fp);

        }

        // Increasing number of objects
        if (strtok(line, " \t\n") != 0)
            (*n_objs)++;

    } // while

    rewind(fp);

    // Finding number of coords
    (*n_coords) = 0;
    while (fgets(line, line_len, fp) != NULL) {

        if (strtok(line, " \t\n") != 0) {

            // Ignoring the id
            while (strtok(NULL, " ,\t\n") != NULL)
                (*n_coords)++;

            break; 
        }

    } // while

    rewind(fp);

    // Allocating space for objects[][]
    len = (*n_objs) * (*n_coords);
    objects = (float**) malloc((*n_objs) * sizeof(float*));
    objects[0] = (float*) malloc(len * sizeof(float));

    for (i=1; i<(*n_objs); i++)
        objects[i] = objects[i-1] + (*n_coords);

    i = 0;

    // Reading objects
    while (fgets(line, line_len, fp) != NULL) {

        if (strtok(line, " \t\n") == NULL)
            continue;

        // Converting strings into floats
        for (j=0; j<(*n_coords); j++)
            objects[i][j] = atof(strtok(NULL, " ,\t\n"));
        
        i++;
    }

    // Closing files and memory free
    fclose(fp);
    free(line);

    return objects;

} // end file_read


// Utility function to write results in two distinct files
int file_write(char *filename, int n_clusters, int n_objs, int n_coords, float **clusters, int *membership)
{
    FILE *fp;
    int i, j;
    char output_filename[1024];

    // Writing coordinates of the cluster centres 
    sprintf(output_filename, "cluster_centres.csv");
    
    fp = fopen(output_filename, "w");
    for (i=0; i<n_clusters; i++) {
        fprintf(fp, "%d ", i);
        for (j=0; j<n_coords; j++)
            fprintf(fp, "%f ", clusters[i][j]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    // Writing cluster results (one line for each data point)
    sprintf(output_filename, "membership.csv");
    
    fp = fopen(output_filename, "w");
    for (i=0; i<n_objs; i++)
        fprintf(fp, "%d\n", membership[i]);
    fclose(fp);

    return 1;
}