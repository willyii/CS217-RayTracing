#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <algorithm>
#include "vec.h"

typedef unsigned int Pixel;

inline Pixel Pixel_Color(const vec3& color)
{
    unsigned int r=std::min(color[0],1.0)*255;
    unsigned int g=std::min(color[1],1.0)*255;
    unsigned int b=std::min(color[2],1.0)*255;
    return (r<<24)|(g<<16)|(b<<8)|0xff;
}

inline vec3 From_Pixel(Pixel color)
{
    return vec3(color>>24,(color>>16)&0xff,(color>>8)&0xff)/255.;
}

class Camera
{
public:
    // Describes camera in space
    vec3 position; // camera position
    vec3 film_position; // where (0.0, 0.0) in the image plane is located in space
    vec3 look_vector; // points from the position to the focal point - normalized
    vec3 vertical_vector; // point up in the image plane - normalized
    vec3 horizontal_vector; // points to the right on the image plane - normalized

    // Describes the coordinate system of the image plane
    vec2 min,max; // coordinates of film corners: min = (left,bottom), max = (right,top)
    vec2 image_size; // physical dimensions of film
    vec2 pixel_size; // physical dimensions of a pixel

    // Describes the pixels of the image
    ivec2 number_pixels; // number of pixels: x and y direction
    Pixel* colors; // Pixel data; row-major order
    
    Camera();
    ~Camera();

    // Used for setting up camera parameters
    void Position_And_Aim_Camera(const vec3& position_input,
        const vec3& look_at_point,const vec3& pseudo_up_vector);
    void Focus_Camera(double focal_distance,double aspect_ratio,
        double field_of_view);
    void Set_Resolution(const ivec2& number_pixels_input);

    // Used for determining the where pixels are
    vec3 World_Position(const ivec2& pixel_index);
    vec2 Cell_Center(const ivec2& index) const
    {
        return min+(vec2(index)+vec2(.5,.5))*pixel_size;
    }

    // Call to set the color of a pixel
    void Set_Pixel(const ivec2& pixel_index,const Pixel& color)
    {
        int i=pixel_index[0];
        int j=pixel_index[1];
        colors[j*number_pixels[0]+i]=color;
    }
};
#endif
