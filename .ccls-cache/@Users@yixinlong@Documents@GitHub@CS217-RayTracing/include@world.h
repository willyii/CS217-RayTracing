#ifndef __WORLD_H__
#define __WORLD_H__

#include <vector>

#include "camera.h"
#include "object.h"

class Ray;

class World {
public:
  Camera camera;
  std::vector<Object *> object_list;
  vec3 bachground_color;
}

#endif
