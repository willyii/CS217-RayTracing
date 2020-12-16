#ifndef __SPHERE_H__
#define __SPHERE_H__

#include "object.h"

class Sphere : public Object {
public:
  __device__ __host__ Sphere(){};
  __device__ __host__ Sphere(vec3 c, double r, vec3 clr) : center(c), radius(r) { color = clr; }
  __device__ __host__ virtual void Intersection(Ray &ray, Hit &hit) const;
  __device__ __host__ virtual vec3 Norm(vec3 &point) const;
  vec3 center;
  double radius;
};

__device__ __host__ void Sphere::Intersection(Ray &ray, Hit &hit) const {
  vec3 oc = ray.endPoint - this->center;
  double a = dot(ray.direction, ray.direction);
  double b = dot(oc, ray.direction);
  double c = dot(oc, oc) - radius * radius;
  double discriminant = b * b - a * c;
  if (discriminant >= 0) {
    /* Small root */
    double d = (-b - sqrt(discriminant)) / a;
    if (d > small_t && d< hit.dist) {
      hit.object = this;
      hit.dist   = d;
      return;
    }
    /* Large root */
    d = (-b + sqrt(discriminant)) / a;
    if (d > small_t && d< hit.dist) {
      hit.object = this;
      hit.dist   = d;
      return;
    }
  }
  return;
}

__device__ __host__ vec3 Sphere::Norm(vec3 &point) const {
  return (point- center).normalized();
}


#endif
