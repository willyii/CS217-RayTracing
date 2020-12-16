#ifndef __PLANE_H__
#define __PLANE_H__

class Plane: public Object{

public:
    __device__  Plane(){};
    __device__  Plane(vec3 x1_in, vec3 norm_in, Shader **shader_list, int idx)
     : x1(x1_in), norm(norm_in){ shader = shader_list[idx];}
    __device__  virtual vec3 Norm(vec3 &point) const {return norm;}
    __device__  virtual void Intersection(Ray &ray, Hit &hit) const;

    vec3 x1;
    vec3 norm;    
};

__device__  
void Plane::Intersection(Ray &ray, Hit &hit) const {

    double dist=(dot(x1,norm)-dot(ray.endPoint,norm))/dot(ray.direction,norm);
    if(dist>small_t && dist < hit.dist)
    {
        hit.dist=dist;
        hit.object=this;
    }
}




#endif