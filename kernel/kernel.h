#include "sphere.h"
#include "camera.h"

/* Add Spheres */
__global__ 
void addSphere(Object **objs){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(objs) = new Sphere(vec3(0.0, 0.0, 99.0), 3,  vec3(0.3, 0.3, 0.3));
        *(objs + 1) = new Sphere(vec3(0.0, 0.0, 150.0), 10, vec3(0.2, 0.7, 0.1));
    }
}

/* Create Camera in GPU */
__global__ 
void creatCamera(Camera **camera, vec3 pos, vec3 look, vec3 up, int width, int height){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        camera[0] = new Camera();
        const double pi = 4 * std::atan(1.0);
        camera[0]->Position_And_Aim_Camera(pos, look, up);
        camera[0]->Focus_Camera(1.0, (double)width / height, 30.0 * (pi / 180));
        camera[0]->Set_Resolution(width, height);
    }
}

/* Convert color vector from double to int */
__device__ 
void vecToColor(ivec3 *color, int idx , vec3 &dcolor){
    color[idx][0] = 255 * dcolor[0];
    color[idx][1] = 255 * dcolor[1];
    color[idx][2] = 255 * dcolor[2];
    return;
}

/* Color single ray */
__device__ 
void castRay(Object **objs, Ray &ray, Hit *result, int N){
    for(int i=0;i<N;i++){
        objs[i]->Intersection(ray, result);
    }
    return;
}

/* Render */
__global__ 
void render(Object **objs, int N,  ivec3 *colors, int width, int height, 
            Camera **camera, vec3 *testVec){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i>= width || j >= height) return;
    int color_index = j*width + i;
    
    /* Initalize the ray */
    vec3 endpoint = camera[0]->position;
    vec3 direction= camera[0]->World_Position(ivec2(i,j)) - endpoint;
    vec3 dcolor   = vec3(0.0, 0.0, 0.0);
    Ray  ray      = Ray(endpoint, direction);
    Hit result    = {NULL, __DBL_MAX__, vec3(0.0,0.0,0.0)};

    /* Cast Ray and Set color */
    castRay(objs, ray, &result, N);
    if(result.dist < __DBL_MAX__){ // hit something        
        dcolor = result.object->color;
    }
    else{
        vec3 unit_direct = ray.direction;
        double t = 0.5f * (unit_direct[1] + 1.0f);
        dcolor = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
    vecToColor(colors, color_index, dcolor);
    return;
}
