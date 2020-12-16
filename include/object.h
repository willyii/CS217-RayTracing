#ifndef __OBJECT_H__
#define __OBJECT_H__

#include "ray.h"

struct Hit {
  const Object *object;
  double dist;
};

class Shader;

class Object {
public:
  __device__  Object(){};
  __device__  ~Object(){};
  __device__  virtual void Intersection(Ray &ray, Hit &hit) const = 0;
  __device__  virtual vec3 Norm(vec3 &point) const = 0;

  Shader *shader;
  static const double small_t = 1e-4;
};

#endif
