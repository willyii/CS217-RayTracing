#include <iostream>

#include "phong_shader.h"
#include "point_light.h"
#include "sphere.h"
#include "plane.h"
#include "util.h"
#include "world.h"
#include "color.h"
#include <time.h>

int main() {
  clock_t start,end;
  start=clock();
  /* Set Camera */
  World world = World();
  vec3 postion(0.0, 1.0, 6.0);
  vec3 look_at_point(0.0, 0.0, 0.0);
  vec3 pseudo_up_vector(0.0, 1.0, 0.0);
  world.camera.Position_And_Aim_Camera(postion, look_at_point,
                                       pseudo_up_vector);
  world.camera.Focus_Camera(1.0, (double)width / height, 70.0 * (pi / 180));
  world.camera.Set_Resolution(ivec2(width, height));
  world.background_color = vec3(0.0, 0.0, 0.0);

  double specular_power=50;
  
  Shader *shader_red=new Phong_Shader(world,red,red,white,specular_power);
  Shader *shader_blue=new Phong_Shader(world,blue,blue,white,specular_power);
  Shader *shader_gray=new Phong_Shader(world,gray,gray,white,specular_power);
  Shader *shader_green=new Phong_Shader(world,green,green,white,specular_power);
  Shader *shader_white=new Phong_Shader(world,white,white,white,specular_power);
  Shader *shader_magenta=new Phong_Shader(world,magenta,magenta,white,specular_power);

  /* Set Point_light */
  world.lights.push_back(new Point_Light(vec3(0,5,6),white,300));
  world.lights.push_back(new Point_Light(vec3(-4,2,6),white,300));
  world.lights.push_back(new Point_Light(vec3(0,-3,6),green,10));


  /* TODO: Set Objects */
  world.object_list.push_back(new Plane(vec3(0, -2, 0), vec3(0,1,0), shader_gray));
  world.object_list.push_back(new Sphere(vec3(1, 0, 0), 0.5, shader_red));
  world.object_list.push_back(new Sphere(vec3(0, 0, 1), 0.5, shader_blue));
  
  world.Render();
  end=clock();
  std::cout<<"Running Time: "<<(double)(end-start)/CLOCKS_PER_SEC<<" s"<<std::endl;
  /* TODO: Save Image */
  Dump_png(world.camera.colors, width, height, "./result/scene2.jpg");

  return 0;
}
