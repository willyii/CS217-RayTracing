#ifndef __WORLD_H__
#define __WORLD_H__

#include <algorithm>
#include <vector>

#include "camera.h"
#include "object.h"

class Ray;

__global__
void Render(Camera *camera, Object *obj, int N, int *pic, int width , int height){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
}

class World {
public:
    __device__ World();
    __device__ ~World();

    Camera camera;
    std::vector<Object *> object_list;
    vec3 background_color;
    __device__ void Render();

    __device__ void RenderSinglePixel(const ivec2 &index);

    __device__ vec3 CastRay(const Ray &ray);
};

__device__ World::World() : background_color(0) {}

__device__ World::~World() {
  for (size_t i = 0; i < object_list.size(); i++)
    delete object_list[i];
}

__device__ void World::RenderSinglePixel(const ivec2 &index) {
  vec3 endpoint = camera.position;
  vec3 direction = camera.World_Position(index) - endpoint;
  Ray ray = Ray(endpoint, direction);

  vec3 color = CastRay(ray);

  camera.Set_Pixel(index, Pixel_Color(color));
  return;
}

__device__ World::CastRay(const Ray &ray) {
  // std::cout << "Current ray start from " << ray.endPoint << " to "
  //          << ray.direction << std::endl;
  vec3 color = background_color;
  Hit closest = {NULL, __DBL_MAX__, vec3(0.0, 0.0, 0.0)};
  Hit currentHit;
  for (size_t i = 0; i < object_list.size(); i++) {
    currentHit = object_list[i]->Intersection(ray);
    if (currentHit.dist == __DBL_MAX__)
      continue;
    if (currentHit.dist < closest.dist)
      closest = currentHit;
  }
  if (closest.dist == __DBL_MAX__) {
    return background_color;
  }
  return closest.object->color;
}

#endif
