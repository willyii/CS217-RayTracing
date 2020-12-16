#ifndef __POINT_LIGHT_H__
#define __POINT_LIGHT_H__

#include "light.h"

class Point_Light : public Light
{
public:
    __device__ 
    Point_Light(const vec3& position,const vec3& color,double brightness)
        :Light(position,color,brightness)
    {}

    __device__ 
    vec3 Emitted_Light(const vec3& vector_to_light) const
    {
        const double pi = 4 * std::atan(1.0);
        return color*brightness/(4*pi*vector_to_light.magnitude_squared());
    }
};
#endif
