#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
#include "util.h"
#include "shader.h"
#include "color.h"
#include <time.h>

int main(int argc, char *argv[]) {
    clock_t start, end;
    start = clock();

    /* Camera parameter   */
    vec3 pos    = vec3(0.0, 1.0, 6.0);
    vec3 look   = vec3(0.0, 0.0, 0.0);
    vec3 up     = vec3(0.0, 1.0, 0.0);
    double phi  = 70.0;

    /* Initialize camera  */
    Camera **camera;
    checkCudaErrors(cudaMalloc((void **)&camera, sizeof(Camera *)));
    creatCamera<<<1,1>>>(camera, pos, look, up, width, height, phi);
    checkCudaErrors(cudaDeviceSynchronize());

    /* Initialize Light Sources */
    int N_light = 2;
    Light **lights;
    checkCudaErrors(cudaMalloc((void **)&lights, sizeof(Light *) * N_light));
    createLight<<<1,1>>>(lights, 0, vec3(0,4,6), white, 200); // TODO: parse from file 
    createLight<<<1,1>>>(lights, 1, vec3(-3,1,6), magenta, 200); // TODO: parse from file 

    /* Initialize Shader */
    int N_shader = 5;
    Shader **shaders;
    checkCudaErrors(cudaMalloc((void **)&shaders, sizeof(Shader *) * N_shader));
    createShader<<<1,1>>>(shaders,0, red, red, white, 50); // red
    createShader<<<1,1>>>(shaders,1, green, green,white, 50); // green
    createShader<<<1,1>>>(shaders,2, blue, blue, white, 50); // blue
    createShader<<<1,1>>>(shaders,3, white, white,white, 50); // white
    createShader<<<1,1>>>(shaders,4, gray, gray,white, 50); // gray
    
    /* Initialize objects */
    int N_objs = 5; // number of objects
    Object **objs;
    checkCudaErrors(cudaMalloc((void **)&objs, sizeof(Object *) * N_objs));
    addPlane<<<1,1>>>(objs, 0, shaders, 4, vec3(0,-1,0), vec3(0,1,0)); 
    addSphere<<<1,1>>>(objs, 1, shaders, 3, vec3(0,0,0), 0.5); 
    addSphere<<<1,1>>>(objs, 2, shaders, 0, vec3(1,0,0), 0.5); 
    addSphere<<<1,1>>>(objs, 3, shaders, 1, vec3(0,1,0), 0.5); 
    addSphere<<<1,1>>>(objs, 4, shaders, 2, vec3(0,0,1), 0.5); 
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
    dim3 blocks(width/tx+1, height/ty +1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(world, colors, width, height);
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();

    printf("Render time: %f s\n", double(end - start)/CLOCKS_PER_SEC);

    // Save img
    Dump_png(colors, width, height, "./result/scene1.png");


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
