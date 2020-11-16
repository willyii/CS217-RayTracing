#ifndef __SPHERE_H__
#define __SPHERE_H__

#include "object.h"

class Sphere : public Object {
  Sphere(){};
  Sphere(vec3 c, double r) : center(c), radius(r) {}
  virtual void Intersection(const Ray &ray, double t_min, double t_max,
                            Hit &hit) const;
  vec3 center;
  double radius;
};

void Sphere::Intersection(const Ray &ray, double t_min, double t_max,
                          Hit &hit) const {
  return;
}

#endif
