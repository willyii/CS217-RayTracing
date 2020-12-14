#include <stdio.h>
#include <stdlib.h>
#include "vec.h"
#include "testKernel.h"




int main(int argc, char *argv[]) {

  testVec3();

  testRay();

  testCamera();

  testRender();

  




  // /* Set Camera */
  // int width = 640;
  // int height = 480;
  // World world = World();
  // vec3 postion(0.0, 0.0, 0.0);
  // vec3 look_at_point(0.0, 0.0, -1.0);
  // vec3 pseudo_up_vector(0.0, 1.0, 0.0);
  // world.camera.Position_And_Aim_Camera(postion, look_at_point,
  //                                      pseudo_up_vector);
  // world.camera.Focus_Camera(1.0, (double)width / height, 30.0 * (pi / 180));
  // world.camera.Set_Resolution(ivec2(width, height));
  // world.background_color = vec3(0.0, 1.0, 0.0);

  // /* TODO: Set Objects */
  // vec3 center(0.0, 0.0, -10.0);
  // double radius = 0.5;
  // vec3 color(.2, 0.2, .8);
  // Sphere *s1 = new Sphere(center, radius, color);
  // world.object_list.push_back(s1);

  // world.Render();

  // /* TODO: Save Image */
  // Dump_png(world.camera.colors, width, height, "./result/test.jpg");

  // ivec2 *resulotuion = 

  return 0;
}
