#ifndef __OBJECT_H__
#define __OBJECT_H__

#include "ray.h"

struct Hit {
  const Object *object;
  double dist;
};

class Object {
public:
  __device__ __host__ Object(){};
  __device__ __host__ ~Object(){};
  __device__ __host__ virtual void Intersection(Ray &ray, Hit &hit) const = 0;
  __device__ __host__ virtual vec3 Norm(vec3 &point) const = 0;

  vec3 color;
  static const double small_t = 1e-4;
};

#endif
