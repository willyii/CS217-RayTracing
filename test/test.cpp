#include <iostream>

#include "sphere.h"
#include "util.h"
#include "world.h"
#include "phong_shader.h"
#include "point_light.h"
int main() {
  /* Set Camera */
  int width = 640;
  int height = 480;
  World world = World();
  vec3 postion(0.0, 0.0, 0.0);
  vec3 look_at_point(0.0, 0.0, -1.0);
  vec3 pseudo_up_vector(0.0, 1.0, 0.0);
  world.camera.Position_And_Aim_Camera(postion, look_at_point,
                                       pseudo_up_vector);
  world.camera.Focus_Camera(1.0, (double)width / height, 30.0 * (pi / 180));
  world.camera.Set_Resolution(ivec2(width, height));
  world.background_color = vec3(0.0, 1.0, 0.0);

  vec3 amb_color(0.0,0.0,0.0);
  vec3 diff_color(1.0,1.0,1.0);
  vec3 spec_color(1.0,1.0,1.0);
  double specular_power=50;
  
  Shader *shader=new Phong_Shader(world,amb_color,diff_color,spec_color,specular_power);

  /* Set Point_light */
  vec3 light_color(1.0,1.0,1.0);
  vec3 light_position(0.0,1.0,10.0);
  double brightness=200;
  world.lights.push_back(new Point_Light(light_position,light_color,brightness));
  /* TODO: Set Objects */
  vec3 center(0.0, 0.0, -10.0);
  double radius = 0.5;
  vec3 color(.2, 0.2, .8);
  Sphere *s1 = new Sphere(center, radius, color);
  s1->material_shader=shader;
  world.object_list.push_back(s1);

  world.Render();

  /* TODO: Save Image */
  Dump_png(world.camera.colors, width, height, "./result/test.jpg");

  return 0;
}
