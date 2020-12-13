#ifndef __UTIL_H__
#define __UTIL_H__

//#include <png.h>

#include <cassert>
#include <fstream>
#include <iostream>

/**
 * Save image to file.
 * 
 * data: ivec3, save color for each pixel, row major
 */
void Dump_png(ivec3 *data, int width, int height, const char *filename) {
  std::ofstream WriteFile;
  ivec3 current;
  WriteFile.open(filename, std::ios::trunc);
  WriteFile << "P3\n" << width << " " << height << "\n255\n";
  for (size_t j = 0; j < height; j++) {
    for (size_t i = 0; i < width; i++) {
      current = data[j * width + i];
      WriteFile << current[0]<< " " << current[1] << " "
                << current[2] << "\n";
    }
  }
  WriteFile.close();
  return;
}
#endif
