#ifndef __PHONG_SHADER_H__
#define __PHONG_SHADER_H__

#include "shader.h"
#include "ray.h"
#include "world.h"
#include "object.h"
#include "light.h"

class Phong_Shader : public Shader
{
public:
    vec3 color_ambient,color_diffuse,color_specular;
    double specular_power;

    Phong_Shader(World& world_input,
        const vec3& color_ambient,
        const vec3& color_diffuse,
        const vec3& color_specular,
        double specular_power)
        :Shader(world_input),color_ambient(color_ambient),
        color_diffuse(color_diffuse),color_specular(color_specular),
        specular_power(specular_power)
    {}

    virtual vec3 Shade_Surface(const Ray& ray,const vec3& intersection_point,
        const vec3& normal,int recursion_depth) const override;
};

vec3 Phong_Shader::
Shade_Surface(const Ray& ray,const vec3& intersection_point,const vec3& normal,int recursion_depth) const
{
    vec3 color={0,0,0};
    vec3 light_endpoint=intersection_point;
    vec3 diffuse={0,0,0};
    vec3 specular={0,0,0};
    vec3 ambient= color_ambient *world.ambient_color*world.ambient_intensity;    
    double nl;
    double rv;
    for(int i=0;i<world.lights.size();i++)
    {
        
        vec3 vector_to_light=(world.lights[i]->position - intersection_point);
        vec3 light_direction=vector_to_light.normalized();
        vec3 n=normal.normalized();
        vec3 v=(ray.endPoint-intersection_point).normalized();
        vec3 r=(2*dot(n,light_direction)*n-light_direction).normalized();
        Ray light(light_endpoint,light_direction);
        double distance_to_light=vector_to_light.magnitude();
        if(dot(n,light_direction)>0) 
            nl=dot(n,light_direction);
        else nl=0;

        if(dot(r,v)>0)
            rv=dot(r,v);
        else rv=0;

        if(world.enable_shadows)
        {
            if(world.Closest_Intersection(light).dist>distance_to_light)
            {
                diffuse += color_diffuse * world.lights[i]->Emitted_Light(vector_to_light) * nl;
                specular += color_specular * world.lights[i]->Emitted_Light(vector_to_light)* pow(rv,specular_power);       
            }
        }
        else 
        {
            diffuse += color_diffuse * world.lights[i]->Emitted_Light(vector_to_light) * nl;
            specular += color_specular * world.lights[i]->Emitted_Light(vector_to_light)* pow(rv,specular_power); 
        }
        
    }
    color=ambient+diffuse+specular;
    return color;

}
#endif
