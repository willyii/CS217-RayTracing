#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <algorithm>

#include "vec.h"

class Camera {
public:
  // Describes camera in space
  vec3 position;      // camera position
  vec3 film_position; // where (0.0, 0.0) in the image plane is located in space
  vec3 look_vector; // points from the position to the focal point - normalized
  vec3 vertical_vector;   // point up in the image plane - normalized
  vec3 horizontal_vector; // points to the right on the image plane - normalized

  // Describes the coordinate system of the image plane
  vec2 min, max;   // coordinates of film corners: min = (left,bottom), max =
                   // (right,top)
  vec2 image_size; // physical dimensions of film
  vec2 pixel_size; // physical dimensions of a pixel

  // Describes the pixels of the image
  ivec2 number_pixels; // number of pixels: x and y direction

  __device__ __host__
  Camera(){};

  // Used for setting up camera parameters
  __device__ __host__
  void Position_And_Aim_Camera(const vec3 &position_input,
                               const vec3 &look_at_point,
                               const vec3 &pseudo_up_vector);

  __device__ __host__
  void Focus_Camera(double focal_distance, double aspect_ratio,
                    double field_of_view);

  __device__ __host__
  void Set_Resolution(const int width, const int height);

  // Used for determining the where pixels are
  __device__ __host__
  vec3 World_Position(const ivec2 pixel_index);

  __device__ __host__
  vec2 Cell_Center(const ivec2 index) const {
    return min + (vec2(index) + vec2(.5, .5)) * pixel_size;
  }
};

__device__ __host__
void Camera::Position_And_Aim_Camera(const vec3 &position_input,
                                     const vec3 &look_at_point,
                                     const vec3 &pseudo_up_vector) {
  position = position_input;
  look_vector = ((look_at_point) - position).normalized();
  horizontal_vector = cross(look_vector, pseudo_up_vector).normalized();
  vertical_vector = cross(horizontal_vector, look_vector).normalized();
}

__device__ __host__
void Camera::Set_Resolution(const int width, const int height) {
  number_pixels = ivec2(width, height);
  min = -0.5 * image_size;
  max = 0.5 * image_size;
  pixel_size = image_size / vec2(number_pixels);
}

// Find the world position of the input pixel
__device__ __host__
vec3 Camera::World_Position(const ivec2 pixel_index) {
  vec3 result;
  vec2 cell_center = Cell_Center(pixel_index);
  result = film_position + cell_center[0] * horizontal_vector +
           cell_center[1] * vertical_vector;
  return result;
}

// Maybe use later
__device__ __host__
void Camera::Focus_Camera(double focal_distance, double aspect_ratio,
                          double field_of_view) {
  film_position = position + look_vector * focal_distance;
  double width = 2.0 * focal_distance * tan(.5 * field_of_view);
  double height = width / aspect_ratio;
  image_size = vec2(width, height);
}
#endif
