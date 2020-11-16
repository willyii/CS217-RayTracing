#include <iostream>

#include "sphere.h"
#include "util.h"
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
  world.background_color = vec3(1.0, 1.0, 1.0);

  /* TODO: Set Objects */
  vec3 center(0.0, 0.0, -10);
  double radius = 3;
  vec3 color(0.0, 0.0, 0.0);
  Sphere *s1 = new Sphere(center, radius, color);
  world.object_list.emplace_back(s1);

  world.Render();

  Dump_png(world.camera.colors, 800, 600, "test.png");

  // std::cout << "This is  Test Program" << std::endl;
  return 0;
}
