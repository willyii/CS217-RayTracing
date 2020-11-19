#ifndef __UTIL_H__
#define __UTIL_H__

//#include <png.h>

#include <cassert>
#include <fstream>
#include <iostream>

typedef unsigned int Pixel;

// void Dump_png(Pixel *data, int width, int height, const char *filename) {
//  FILE *file = fopen(filename, "wb");
//  assert(file);

//  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0,
//  0); assert(png_ptr); png_infop info_ptr = png_create_info_struct(png_ptr);
//  assert(info_ptr);
//  bool result = setjmp(png_jmpbuf(png_ptr));
//  assert(!result);
//  png_init_io(png_ptr, file);
//  int color_type = PNG_COLOR_TYPE_RGBA;
//  png_set_IHDR(png_ptr, info_ptr, width, height, 8, color_type,
//               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
//               PNG_FILTER_TYPE_DEFAULT);

//  Pixel **row_pointers = new Pixel *[height];
//  for (int j = 0; j < height; j++)
//    row_pointers[j] = data + width * (height - j - 1);
//  png_set_rows(png_ptr, info_ptr, (png_byte **)row_pointers);
//  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_BGR |
//  PNG_TRANSFORM_SWAP_ALPHA,
//                0);
//  delete[] row_pointers;
//  png_destroy_write_struct(&png_ptr, &info_ptr);
//  fclose(file);
//}

inline vec3 From_Pixel(Pixel color) {
  return vec3(color >> 24, (color >> 16) & 0xff, (color >> 8) & 0xff) / 255.;
}

void Dump_png(Pixel *data, int width, int height, const char *filename) {
  std::ofstream WriteFile;
  vec3 current;
  WriteFile.open(filename, std::ios::trunc);
  WriteFile << "P3\n" << width << " " << height << "\n255\n";
  for (size_t j = 0; j < height; j++) {
    for (size_t i = 0; i < width; i++) {
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
