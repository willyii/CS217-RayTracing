#ifndef __PHONG_SHADER_H__
#define __PHONG_SHADER_H__

#include "shader.h"

class Phong_Shader : public Shader
{
public:
    vec3 color_ambient,color_diffuse,color_specular;
    double specular_power;

    __device__
    Phong_Shader(
        const vec3 color_ambient,
        const vec3 color_diffuse,
        const vec3 color_specular,
        double specular_power)
        :color_ambient(color_ambient),
        color_diffuse(color_diffuse),color_specular(color_specular),
        specular_power(specular_power)
    {}

    __device__
    virtual vec3 Shade_Surface(const Ray ray,const vec3 intersection_point,
        const vec3 normal, World **world) const;
};

__device__
vec3 Phong_Shader::
Shade_Surface(const Ray ray,const vec3 intersection_point,
        const vec3 normal, World **world) const
{
    vec3 diffuse(0,0,0);
    vec3 specular(0,0,0);
    vec3 ambient= color_ambient * (*world)->ambient_color * (*world)->ambient_intensity;    
    double nl;
    double rv;
    for(int i=0;i<(*world)->num_lights;i++)
    {
        vec3 vector_to_light=((*world)->light_list[i]->position - intersection_point);
        Ray light(intersection_point,vector_to_light);
        vec3 n=normal.normalized();
        vec3 v=(ray.endPoint-intersection_point).normalized();
        vec3 r=(2*dot(n,light.direction)*n-light.direction).normalized();
        double distance_to_light=vector_to_light.magnitude();

        nl = dot(n,light.direction);
        if(nl < 0) nl = 0.0;

        rv = dot(r, v);
        if(rv<0) rv = 0.0;

        Hit result = {NULL, __DBL_MAX__};
        (*world)->Closest_Intersection(light, result);     
        if(result.dist>distance_to_light)
        {
            diffuse += color_diffuse * (*world)->light_list[i]->Emitted_Light(vector_to_light) * nl;
            specular += color_specular * (*world)->light_list[i]->Emitted_Light(vector_to_light)* pow(rv,specular_power);       
        }
    }
    return ambient+diffuse+specular;

}
#endif