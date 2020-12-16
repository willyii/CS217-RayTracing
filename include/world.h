#ifndef __WORLD_H__
#define __WORLD_H__

#include "camera.h"
#include "object.h"
#include "light.h"


class World {
public:

  __device__
  World(Camera **camera, Object **objs, int N_objs, Light **lights, int N_lights, vec3 back, vec3 ambient, int intense);

  __device__
  ~World();

  __device__
  void Closest_Intersection(Ray ray, Hit &hit);

  Camera **camera;
  Object **object_list;
  Light  **light_list;
  vec3 background_color;
  vec3 ambient_color;
  double ambient_intensity;
  int num_objs;
  int num_lights;

};

__device__
World::World(Camera **camera, Object **objs, int N_objs, Light **lights, int N_lights,vec3 back, vec3 ambient, int intense)
: camera(camera), object_list(objs), num_objs(N_objs), background_color(back), 
  ambient_color(ambient), ambient_intensity(intense), light_list(lights), num_lights(N_lights)
  {}

__device__
void World::Closest_Intersection(Ray ray, Hit &hit){
    for(int i=0;i<num_objs;i++){
        object_list[i]->Intersection(ray, hit);
    }
}

#endif