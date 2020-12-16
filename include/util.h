#ifndef __UTIL_H__
#define __UTIL_H__

//#include <png.h>

#include <cassert>
#include <fstream>
#include <iostream>

typedef unsigned int Pixel;

inline vec3 From_Pixel(Pixel color) {
  return vec3(color >> 24, (color >> 16) & 0xff, (color >> 8) & 0xff) / 255.;
}

void Dump_png(Pixel *data, int width, int height, const char *filename) {
  std::ofstream WriteFile;
  vec3 current;
  WriteFile.open(filename, std::ios::trunc);
  WriteFile << "P3\n" << width << " " << height << "\n255\n";
  for (int j = height-1; j >=0; j--) {
    for (int  i = 0; i < width; i++) {
      current = From_Pixel(data[j * width + i]);
      // std::cout << "Test: " << i * width + j << std::endl;
      WriteFile << current[0] * 255 << " " << current[1] * 255 << " "
                << current[2] * 255 << "\n";
      // std::cout << current[0] * 255 << " " << current[1] * 255 << " "
      //<< current[2] * 255 << "\n";
    }
  }
  WriteFile.close();
  return;
}
#endif
