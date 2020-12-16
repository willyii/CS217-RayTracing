#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
#include "util.h"
#include "shader.h"
#include <time.h>

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
    clock_t start, end;
    start = clock();

    int width   = 1920;
    int height  = 1080;
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
    int N_light = 3;
    Light **lights;
    checkCudaErrors(cudaMalloc((void **)&lights, sizeof(Light *) * N_light));
    createLight<<<1,1>>>(lights, 0, vec3(0,5,6), vec3(1,1,1), 400); // TODO: parse from file 
    createLight<<<1,1>>>(lights, 1, vec3(-4,2,6), vec3(1,1,1), 400); // TODO: parse from file 
    createLight<<<1,1>>>(lights, 2, vec3(0,-3,6), vec3(0,1,0), 10); // TODO: parse from file 

    /* Initialize Shader */
    int N_shader = 3;
    Shader **shaders;
    checkCudaErrors(cudaMalloc((void **)&shaders, sizeof(Shader *) * N_shader));
    createShader<<<1,1>>>(shaders,0, vec3(1,0,0), vec3(1,0,0),vec3(1,1,1), 50);
    createShader<<<1,1>>>(shaders,1, vec3(0,0,1), vec3(0,0,1),vec3(1,1,1), 50);
    createShader<<<1,1>>>(shaders,2, vec3(.5,.5,.5), vec3(.5,.5,.5),vec3(1,1,1), 50);


    /* Initialize objects */
    int N_objs = 3; // number of objects
    Object **objs;
    checkCudaErrors(cudaMalloc((void **)&objs, sizeof(Object *) * N_objs));
    addSphere<<<1,1>>>(objs, 0, shaders, 0, vec3(1,0,0), 0.5); 
    addSphere<<<1,1>>>(objs, 1, shaders, 1, vec3(0,0,1), 0.5); 
    addPlane<<<1,1>>>(objs, 2, shaders, 2, vec3(0,-2,0), vec3(0,1,0)); 
    checkCudaErrors(cudaDeviceSynchronize());

    /* Initialize world */
    World **world;
    checkCudaErrors(cudaMalloc((void **)&world, sizeof(World *)));
    createWorld<<<1,1>>>(world, camera, objs, N_objs, lights, N_light, vec3(0,0,0), 0 );

    /* Initialize imgs */
    ivec3 *colors;
    checkCudaErrors(cudaMallocManaged((void **)&colors, sizeof(ivec3)*height*width));
    checkCudaErrors(cudaDeviceSynchronize());

    /* Render            */
    int tx = 8;
    int ty = 8;
    dim3 blocks(width/tx+1, height/ty +1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(world, colors, width, height);
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();

    printf("Render time: %f s\n", double(end - start)/CLOCKS_PER_SEC);

    // Save img
    Dump_png(colors, width, height, "./result/test.png");


    // freeWorld<<<1,1>>>(shaders, N_shader, objs, N_objs, lights, N_light, world);
    checkCudaErrors(cudaFree(shaders));
    checkCudaErrors(cudaFree(objs));
    checkCudaErrors(cudaFree(lights));
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(colors));
    checkCudaErrors(cudaDeviceSynchronize());
    cudaDeviceReset();

    return 0;
}
