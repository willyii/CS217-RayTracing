#ifndef __PLANE_H__
#define __PLANE_H__

#include "object.h"

class Plane : public Object
{
public:
    vec3 x1;
    vec3 normal;

    Plane(const vec3& point,const vec3& normal)
        :x1(point),normal(normal.normalized())
    {}

    virtual Hit Intersection(const Ray& ray) const override;
    virtual vec3 Normal(const vec3& point) const override;
};
#endif
Hit Plane::Intersection(const Ray& ray) const
{
    Hit result;
    double dist;
    vec3 EndPoint=ray.endPoint;
    vec3 Direction=ray.direction;
    dist=(dot(x1,normal)-dot(EndPoint,normal))/dot(Direction,normal);
    if(dist>small_t)
    {
        result.dist=dist;
        result.object=this;
        return result;
    }
    else
    {
        result.dist=INFINITY;
        result.object=nullptr;
        return result;
    }
}

vec3 Plane::Normal(const vec3& point) const
{
    return normal;
}

