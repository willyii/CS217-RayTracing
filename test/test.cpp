#include <iostream>

#include "phong_shader.h"
#include "point_light.h"
#include "sphere.h"
#include "plane.h"
#include "util.h"
#include "world.h"
#include <time.h>
int main() {
  clock_t start,end;
  start=clock();
  /* Set Camera */
  int width = 4096;
  int height = 2160;
  World world = World();
  vec3 postion(0.0, 1.0, 6.0);
  vec3 look_at_point(0.0, 0.0, 0.0);
  vec3 pseudo_up_vector(0.0, 1.0, 0.0);
  world.camera.Position_And_Aim_Camera(postion, look_at_point,
                                       pseudo_up_vector);
  world.camera.Focus_Camera(1.0, (double)width / height, 70.0 * (pi / 180));
  world.camera.Set_Resolution(ivec2(width, height));
  world.background_color = vec3(0.0, 0.0, 0.0);

  vec3 red(1,0,0);
  vec3 green(0,1,0);
  vec3 blue(.2,.2,.8);
  vec3 white(1,1,1);
  vec3 grey(.5,.5,.5);
  double specular_power=50;
  
  Shader *shader_red=new Phong_Shader(world,red,red,white,specular_power);
  Shader *shader_blue=new Phong_Shader(world,blue,blue,white,specular_power);
  Shader *shader_grey=new Phong_Shader(world,grey,grey,white,specular_power);
  /* Set Point_light */
  world.lights.push_back(new Point_Light(vec3(0,5,6),vec3(1,1,1),200));
  world.lights.push_back(new Point_Light(vec3(-4,2,6),vec3(1,1,1),200));
  /* TODO: Set Objects */
  Sphere *s1 = new Sphere(vec3(1, 0, 0), 0.5);
  Sphere *s2 = new Sphere(vec3(0, 0, 1), 0.5);
  Plane *p1 = new Plane(vec3(0,-2,0),vec3(0,1,0));
  s1->material_shader=shader_red;
  s2->material_shader=shader_blue;
  p1->material_shader=shader_grey;
  world.object_list.push_back(s1);
  world.object_list.push_back(s2);
  world.object_list.push_back(p1);
  
  world.Render();
  end=clock();
  std::cout<<"Running Time: "<<(double)(end-start)/CLOCKS_PER_SEC<<" s"<<std::endl;
  /* TODO: Save Image */
  Dump_png(world.camera.colors, width, height, "./result/test.jpg");

  return 0;
}
