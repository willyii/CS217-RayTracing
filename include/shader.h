#ifndef __SHADER_H__
#define __SHADER_H__

#include "ray.h"
#include "world.h"

class Shader
{
public:
    __device__ 
    Shader()
    {}

    __device__ 
    virtual ~Shader()
    {}

    __device__ 
    virtual vec3 Shade_Surface(const Ray ray,const vec3 intersection_point,
        const vec3 normal, World **world) const=0;

};
#endif