#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
#include "util.h"
#include "shader.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

/* Check the error of cuda API */
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


int main(int argc, char *argv[]) {

    /*  Debug thing */
    vec3 *testVec;
    checkCudaErrors(cudaMallocManaged((void **)&testVec, sizeof(vec3)));
    checkCudaErrors(cudaDeviceSynchronize());

    int width   = 640;
    int height  = 480;
    /* Camera parameter   */
    vec3 pos    = vec3(0.0, 1.0, 6.0);
    vec3 look   = vec3(0.0, 0.0, 0.0);
    vec3 up     = vec3(0.0, 1.0, 0.0);

    /* Initialize camera  */
    Camera **camera;
    checkCudaErrors(cudaMalloc((void **)&camera, sizeof(Camera *)));
    creatCamera<<<1,1>>>(camera, pos, look, up, width, height);
    checkCudaErrors(cudaDeviceSynchronize());

    /* Initialize Light Sources */
    int N_light = 2;
    Light **lights;
    checkCudaErrors(cudaMalloc((void **)&lights, sizeof(Light *) * N_light));
    createLight<<<1,1>>>(lights, 0, vec3(0,5,6), vec3(1,1,1), 200, testVec); // TODO: parse from file 
    createLight<<<1,1>>>(lights, 1, vec3(-4,2,6), vec3(1,1,1), 200, testVec); // TODO: parse from file 
    // createLight<<<1,1>>>(lights, 2, vec3(0,-3,6), vec3(0,1,0), 10, testVec); // TODO: parse from file 
    printf("Test Vec3 : %f, %f, %f\n", (*testVec)[0], (*testVec)[1], (*testVec)[2]);

    /* Initialize objects */
    int N_objs = 2; // number of objects
    Object **objs;
    checkCudaErrors(cudaMalloc((void **)&objs, sizeof(Object *) * N_objs));
    addSphere<<<1,1>>>(objs); // TODO: make it flexibale
    checkCudaErrors(cudaDeviceSynchronize());

    /* Initialize world */
    World **world;
    checkCudaErrors(cudaMalloc((void **)&world, sizeof(World *)));
    createWorld<<<1,1>>>(world, camera, objs,N_objs, lights, N_light, vec3(0,0,0), vec3(1,1,1),1, testVec );
    printf("Test Vec3 : %f, %f, %f\n", (*testVec)[0], (*testVec)[1], (*testVec)[2]);


    /* Initialize Shader */
    Shader **shader;
    checkCudaErrors(cudaMalloc((void **)&shader, sizeof(Shader *)));
    createShader<<<1,1>>>(shader, vec3(1,0,0), vec3(1,0,0),vec3(1,1,1), 50, testVec );
    printf("Test Vec3 : %f, %f, %f\n", (*testVec)[0], (*testVec)[1], (*testVec)[2]);

    /* Initialize imgs */
    ivec3 *colors;
    checkCudaErrors(cudaMallocManaged((void **)&colors, sizeof(ivec3)*height*width));
    checkCudaErrors(cudaDeviceSynchronize());

    /* Render            */
    int tx = 8;
    int ty = 8;
    dim3 blocks(width/tx+1, height/ty +1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(shader, world, colors, width, height,testVec);
    checkCudaErrors(cudaDeviceSynchronize());

    // Save img
    Dump_png(colors, width, height, "./result/test.png");
    
    /* Print debug info */
    printf("Color of first pixel: %d, %d, %d\n", colors[0][0], colors[0][1], colors[0][2]);
    printf("Test Vec3 : %f, %f, %f\n", (*testVec)[0], (*testVec)[1], (*testVec)[2]);


  return 0;
}
