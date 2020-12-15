#ifndef __OBJECT_H__
#define __OBJECT_H__

#include "ray.h"

struct Hit {
  const Object *object;
  double dist;
  vec3 normal;
};

class Object {
public:
  __device__ __host__ Object(){};
  __device__ __host__ ~Object(){};
  __device__ __host__ virtual void Intersection(Ray &ray, Hit *hit) const = 0;
  vec3 color;
};

#endif
