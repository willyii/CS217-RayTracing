#ifndef __RAY_H__
#define __RAY_H__

#include "vec.h"

class Object;
class RAY
{
public:
    vec3 endPoint;  //endpoint of ray where t=0
    vec3 direction;  // direction of the ray - unit vector

    Ray()
        :endPoint(0,0,0),diretion(0,0,1)
    {}
    Ray(const vec3& endpoint_input,const vec3& direction_input)
        :endPoint(endpoint_input),direction(direction_input.normalized())
    {}

    vec3 Point(double t) const
    {
        return endpoint+direction*t;
    }

};
#endif