#ifndef _H_KMEANS
#define _H_KMEANS

extern int _debug;

float** file_read(char*, int*, int*);
int file_write(char*, int, int, int, float**, int*);
float** omp_kmeans(int, float**, int, int, int, float, int*);

#endif