#ifndef __WORLD_H__
#define __WORLD_H__

#include <vector>

#include "camera.h"
#include "object.h"

class Ray;

class World {
public:
  World();
  ~World();

  Camera camera;
  std::vector<Object *> object_list;
  vec3 background_color;

  void Render();

  void RenderSinglePixel(const ivec2 &index);

  vec3 CastRay(const Ray &ray);
};

World::World() : background_color(0) {}

World::~World() {
  for (size_t i = 0; i < object_list.size(); i++)
    delete object_list[i];
}

void World::Render() {
  for (int j = 0; j < camera.number_pixels[1]; j++)
    for (int i = 0; i < camera.number_pixels[0]; i++)
      RenderSinglePixel(ivec2(i, j));
  return;
}

void World::RenderSinglePixel(const ivec2 &index) {
  vec3 endpoint = camera.position;
  vec3 direction = camera.World_Position(index);
  Ray ray = Ray(endpoint, direction);

  vec3 color = CastRay(ray);

  camera.Set_Pixel(index, Pixel_Color(color));
  return;
}

vec3 World::CastRay(const Ray &ray) {
  vec3 color = background_color;
  Hit closest = {nullptr, __DBL_MAX__, vec3(0.0, 0.0, 0.0)};
  Hit currentHit;
  for (size_t i = 0; i < object_list.size(); i++) {
    currentHit = object_list[i]->Intersection(ray);
    if (currentHit.object == nullptr)
      continue;
    if (currentHit.dist < closest.dist)
      closest = currentHit;
  }
  if (closest.object == nullptr)
    return background_color;
  return closest.object->color;
}

#endif
