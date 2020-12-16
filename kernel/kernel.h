#include "sphere.h"
#include "camera.h"
#include "point_light.h"
#include "world.h"
#include "phong.h"

/* Add Spheres */
__global__ 
void addSphere(Object **objs){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(objs) = new Sphere(vec3(1, 0, 0), 0.5, vec3(1, 0, 0));
        *(objs + 1) = new Sphere(vec3(0.0, 0.0, 1.0), 0.5, vec3(0.2, 0.2, 0.8));
    }
}

/* Create Camera in GPU */
__global__ 
void creatCamera(Camera **camera, vec3 pos, vec3 look, vec3 up, int width, int height){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        camera[0] = new Camera();
        const double pi = 4 * std::atan(1.0);
        camera[0]->Position_And_Aim_Camera(pos, look, up);
        camera[0]->Focus_Camera(1.0, (double)width / height, 70.0 * (pi / 180));
        camera[0]->Set_Resolution(width, height);
    }
}

/* Create light sources */
__global__
void createLight(Light **lights, int idx, vec3 pos, vec3 color, double brightness, vec3 *testVec){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(lights + idx) = new Point_Light(pos, color, brightness);
        *testVec  = color;
    }
}

/* Create World in GPU */
__global__ 
void createWorld(World **world, Camera **camera, Object **objs, int N_objs, Light **lights, int N_lights,
                 vec3 back, vec3 ambient, int intense, vec3 *testVec){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(world) = new World(camera, objs, N_objs, lights, N_lights, back, ambient, 1);
        *testVec  = world[0]->object_list[0]->color;
    }
}

__global__
void createShader(Shader **shader, vec3 ambient, vec3 diff, vec3 specular, double power, vec3 *testVec){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(shader) = new Phong_Shader(ambient, diff, specular, power);
        // *testVec  = shader[0]->ttt;
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

/* Render */
__global__ 
void render(Shader **shader, World **world,  ivec3 *colors, int width, int height, vec3 *testVec){
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
    if(result.dist < __DBL_MAX__){ // hit something
        vec3 point = ray.point(result.dist);
        dcolor = (*shader)->Shade_Surface(ray, point, result.object->Norm(point), world );
        if(i == 320 && j == 240) (*testVec) = dcolor;
    }
    else{
        // vec3 unit_direct = ray.direction;
        // double t = 0.5f * (unit_direct[1] + 1.0f);
        // dcolor = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
        
    }
    vecToColor(colors, color_index, dcolor);
    return;
}
