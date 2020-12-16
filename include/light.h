#ifndef __LIGHT_H__
#define __LIGHT_H__

#include <math.h>
#include <vector>
#include "vec.h"
#include "ray.h"

class Light
{
public:
    vec3 position;
    vec3 color; // RGB color components
    double brightness;

    __device__ 
    Light()
        :position(),color(1,1,1),brightness(1)
    {}

    __device__ 
    Light(const vec3& position,const vec3& color,double brightness)
        :position(position),color(color),brightness(brightness)
    {}
    
    __device__ 
    virtual ~Light()
    {}

    __device__ 
    virtual vec3 Emitted_Light(const vec3& vector_to_light) const=0;
};
#endif
