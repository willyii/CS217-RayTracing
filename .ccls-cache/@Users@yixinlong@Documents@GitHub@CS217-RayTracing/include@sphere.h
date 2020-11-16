#ifndef __SPHERE_H__
#define __SPHERE_H__

#include "object.h"

class Sphere : public Object {
public:
  Sphere(){};
  Sphere(vec3 c, double r, vec3 clr) : center(c), radius(r) { color = clr; }
  virtual Hit Intersection(const Ray &ray) const override;
  vec3 center;
  double radius;
};

Hit Sphere::Intersection(const Ray &ray) const {
  Hit ret = {nullptr, 0.0, vec3(0.0, 0.0, 0.0)};
  vec3 oc = ray.endPoint - center;
  double a = dot(ray.direction, ray.direction);
  double b = dot(oc, ray.direction);
  double c = dot(oc, oc) - radius * radius;
  double discriminant = b * b - a * c;
  if (discriminant > 0) {
    /* Small root */
    double tmp = (-b - sqrt(discriminant)) / a;
    if (tmp > 0) {
      ret.object = this;
      ret.dist = tmp;
      ret.normal = (ray.point(tmp) - center) / radius;
      return ret;
    }
    /* Large root */
    tmp = (-b + sqrt(discriminant)) / a;
    if (tmp > 0) {
      ret.object = this;
      ret.dist = tmp;
      ret.normal = (ray.point(tmp) - center) / radius;
      return ret;
    }
  }
  return ret;
}

#endif
