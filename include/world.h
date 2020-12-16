#ifndef __WORLD_H__
#define __WORLD_H__

#include <algorithm>
#include <vector>

#include "camera.h"
#include "object.h"

class Ray;
class Shader;
class Light;

class World {
public:
  World();
  ~World();

  Camera camera;
  std::vector<Object *> object_list;
  std::vector<Light *> lights;
  vec3 background_color;
  vec3 ambient_color;
  double ambient_intensity;

  bool enable_shadows;
  void Render();

  void RenderSinglePixel(const ivec2 &index);
  Hit Closest_Intersection(const Ray &ray);
  vec3 CastRay(const Ray &ray);
};

World::World() 
  : background_color(0),ambient_intensity(0),enable_shadows(true)
  {}

World::~World() {
  for (size_t i = 0; i < object_list.size(); i++) delete object_list[i];
  for (size_t i = 0; i < lights.size(); i++) delete lights[i];
}

void World::Render() {
  for (int j = 0; j < camera.number_pixels[1]; j++)
    for (int i = 0; i < camera.number_pixels[0]; i++)
      RenderSinglePixel(ivec2(i, j));
  return;
}

void World::RenderSinglePixel(const ivec2 &index) {
  vec3 endpoint = camera.position;
  vec3 direction = camera.World_Position(index) - endpoint;
  Ray ray = Ray(endpoint, direction);

  vec3 color = CastRay(ray);
  camera.Set_Pixel(index, Pixel_Color(color));
  return;
}
Hit World::Closest_Intersection(const Ray &ray)
{
  Hit currentHit;
  Hit closest = {NULL, __DBL_MAX__, vec3(0.0, 0.0, 0.0)};
  for (size_t i = 0; i < object_list.size(); i++) 
  {
    currentHit = object_list[i]->Intersection(ray);
    if (currentHit.dist == __DBL_MAX__)
      continue;
    if (currentHit.dist < closest.dist)
      closest = currentHit;
  }
  return closest;
}
vec3 World::CastRay(const Ray &ray) {
  // std::cout << "Current ray start from " << ray.endPoint << " to "
  //          << ray.direction << std::endl;
  vec3 color = background_color;
  Hit closest= Closest_Intersection(ray);
  vec3 point;
  vec3 normal;
  
  if (closest.dist != __DBL_MAX__) 
  {
    point=ray.point(closest.dist);
    normal=closest.object->Normal(point);
    color=closest.object->material_shader->Shade_Surface(ray,point,normal);
  }
  else color=background_color;
  return color;
}

#endif
