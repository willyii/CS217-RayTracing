#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "sphere.h"
#include "camera.h"
#include "point_light.h"
#include "world.h"
#include "phong.h"
#include "plane.h"

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


template<class T>
__device__
T min(T v1, T v2){
    if(v1 < v2) return v1;
    return v2;
}

/* Add Spheres */
__global__ 
void addSphere(Object **objs, int obj_idx, Shader **shaders, int shd_idx, vec3 pos, double radius){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(objs + obj_idx) = new Sphere(pos, radius, shaders, shd_idx);
    }
}

/* Add Plane */
__global__ 
void addPlane(Object **objs, int obj_idx, Shader **shaders, int shd_idx, vec3 x1, vec3 norm){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(objs + obj_idx) = new Plane(x1, norm, shaders, shd_idx);
    }
}

/* Create Camera in GPU */
__global__ 
void creatCamera(Camera **camera, vec3 pos, vec3 look, vec3 up, int width, int height, double phi){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        camera[0] = new Camera();
        const double pi = 4 * std::atan(1.0);
        camera[0]->Position_And_Aim_Camera(pos, look, up);
        camera[0]->Focus_Camera(1.0, (double)width / height, phi * (pi / 180));
        camera[0]->Set_Resolution(width, height);
    }
}

/* Create light sources */
__global__
void createLight(Light **lights, int idx, vec3 pos, vec3 color, double brightness){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(lights + idx) = new Point_Light(pos, color, brightness);
    }
}

/* Create World in GPU */
__global__ 
void createWorld(World **world, Camera **camera, Object **objs, int N_objs, Light **lights, int N_lights,
                vec3 ambient, int intense){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(world) = new World(camera, objs, N_objs, lights, N_lights, ambient, 1);
    }
}

/* Create Shader in GPU */
__global__
void createShader(Shader **shader, int idx, vec3 ambient, vec3 diff, vec3 specular, double power){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(shader + idx) = new Phong_Shader(ambient, diff, specular, power);
    }
}

/* Convert color vector from double to int */
__device__ 
void vecToColor(ivec3 *color, int idx , vec3 &dcolor){
    color[idx][0] = int(255* min(1.0,dcolor[0]));
    color[idx][1] = int(255* min(1.0,dcolor[1]));
    color[idx][2] = int(255* min(1.0,dcolor[2]));
}



// __global__
// void freeWorld(Shader **shaders, int N_shader, Object **objects, int N_objs, Light  **ligths , int N_lights,
//                World  **world)
// {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//     for(int i=0;i< N_shader; i++ ) delete *(shaders + i);
//     for(int i=0;i< N_objs; i++ ) delete *(objects + i);
//     for(int i=0;i< N_lights; i++ ) delete *(ligths + i);
//     delete *(world);
//     }
// }

/* Render */
__global__ 
void render(World **world,  ivec3 *colors, int width, int height){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i>= width || j >= height) return;
    int color_index = j*width + i;
    
    /* Initalize the ray */
    vec3 endpoint = world[0]->camera[0]->position;
    vec3 direction= world[0]->camera[0]->World_Position(ivec2(i,j)) - endpoint;
    vec3 dcolor   = vec3(0.0, 0.0, 0.0);
    Ray  ray      = Ray(endpoint, direction);
    Hit result    = {NULL, __DBL_MAX__};

    /* Cast Ray and Set color */
    (*world)->Closest_Intersection(ray, result);
    if(result.dist < __DBL_MAX__ ){ // hit something
        vec3 point = ray.point(result.dist);
        dcolor = result.object->shader->Shade_Surface(ray, point, result.object->Norm(point), world );
    }
    vecToColor(colors, color_index, dcolor);
    return;
}

#endif