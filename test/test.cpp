#include <iostream>

#include "sphere.h"
#include "world.h"

int main() {
  /* Set Camera */
  World world = World();
  vec3 postion(0.0, 0.0, 0.0);
  vec3 look_at_point(0.0, 0.0, -1.0);
  vec3 pseudo_up_vector(0.0, 1.0, 0.0);
  world.camera.Position_And_Aim_Camera(postion, look_at_point,
                                       pseudo_up_vector);
  world.camera.Set_Resolution(ivec2(800, 600));

  /* TODO: Set Objects */

  std::cout << "This is  Test Program" << std::endl;
  return 0;
}
