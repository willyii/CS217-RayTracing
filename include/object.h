#ifndef __OBJECT_H__
#define __OBJECT_H__

#include "ray.h"

class Shader;
class Object;


struct Hit {
  const Object *object;
  double dist;
  vec3 normal;
};

class Object {
public:
  Shader* material_shader;
  Object():material_shader(0){};
  virtual ~Object(){};
  virtual Hit Intersection(const Ray &ray) const = 0;
  virtual vec3 Normal(const vec3& point) const=0;
  
  vec3 color;
  
};

#endif
