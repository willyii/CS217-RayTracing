#include "camera.h"
#include "ray.h"
#include "util.h"
#include "sphere.h"
#include <assert.h>

__global__
void testVec3Kernel(vec3 *A, double *ans){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockIdx.x * blockDim.x;
    if(idx == 0){
        *ans = (*A).magnitude_squared();
    }
    return;
}

void testVec3(){
    cudaError_t cuda_ret;
    printf("\n Test vec3 on GPU... \n");
    vec3 *A;
    double *ans;

    /*Malloc memory for A in CPU and GPU*/
    cuda_ret = cudaMallocManaged(&A, sizeof(vec3));
    cuda_ret = cudaMallocManaged(&ans, sizeof(double));


    if(cuda_ret != cudaSuccess){
        printf("Unable to malloc memory for vec3\n");
        cudaFree(A);
        cudaFree(ans);
        return;
    }

    *A = vec3(1.0,2.0,3.0);

    testVec3Kernel<<<256, 8>>>(A, ans);
    cuda_ret = cudaDeviceSynchronize();

    if(cuda_ret != cudaSuccess){
        printf("Synchronize failed \n");
        cudaFree(A);
        cudaFree(ans);
        return;
    }

    assert((*ans) == 14.0);

    cudaFree(A);
    cudaFree(ans);

    printf("\n Vec3 test pass...\n");
    printf("===================\n");
    return;
}

__global__
void testRayKernel(vec3 *pos, vec3 *direction, vec3 *point){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockIdx.x * blockDim.x;
    if(idx == 0){
        Ray ray(pos, direction);
        *point = ray.point(3.0);
    }
    return;
}

void testRay(){
    cudaError_t cuda_ret;
    printf("\n Test Ray on GPU... \n");
    vec3 *pos, *direction, *point;

    /*Malloc memory for A in CPU and GPU*/
    cuda_ret = cudaMallocManaged(&pos, sizeof(vec3));
    cuda_ret = cudaMallocManaged(&direction, sizeof(vec3));
    cuda_ret = cudaMallocManaged(&point, sizeof(vec3));

    if(cuda_ret != cudaSuccess){
        printf("Unable to malloc memory for vec3\n");
        cudaFree(pos);
        cudaFree(direction);
        cudaFree(point);
        return;
    }

    *pos = vec3(0.0,0.0,0.0);
    *direction = vec3(1.0, 0.0, 0.0);
    
    testRayKernel<<<256, 8>>>(pos, direction, point);
    cuda_ret = cudaDeviceSynchronize();

    if(cuda_ret != cudaSuccess){
        printf("Synchronize failed \n");
        cudaFree(pos);
        cudaFree(direction);
        cudaFree(point);
        return;
    }

    assert((*point)[0] == 3.0);

    cudaFree(pos);
    cudaFree(direction);
    cudaFree(point);

    printf("\n Ray test pass...\n");
    printf("===================\n");

}

__global__
void testCameraKernel(Camera *camera, int *ans){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockIdx.x * blockDim.x;
    *ans = camera->number_pixels[0] * camera->number_pixels[1];
    return;
}

void testCamera() {
    cudaError_t cuda_ret;
    printf("\n Test Camera on GPU... \n");

    vec3 pos    = vec3(0.0,0.0,0.0);
    vec3 look_at= vec3(0.0, 0.0, 1.0);
    vec3 up_vec = vec3(0.0, 1.0, 0.0);
    int width  = 640;
    int height = 480;
    Camera *camera;
    int *ans;


    /*Malloc memory in CPU and GPU*/
    cuda_ret = cudaMallocManaged(&camera, sizeof(Camera));
    cuda_ret = cudaMallocManaged(&ans, sizeof(int));

    if(cuda_ret != cudaSuccess){
        printf("Unable to malloc memory for parameters\n");
        cudaFree(camera);
        cudaFree(ans);
        return;
    }

    camera->Position_And_Aim_Camera(&pos, &look_at, &up_vec);
    camera->Focus_Camera(1.0, (double)width / height, 30.0 * (pi / 180));
    camera->Set_Resolution(width, height);
    
    testCameraKernel<<<256, 8>>>(camera, ans);
    cuda_ret = cudaDeviceSynchronize();

    if(cuda_ret != cudaSuccess){
        printf("Synchronize failed \n");
        cudaFree(camera);
        cudaFree(ans);
        return;
    }
    assert(*ans == width * height);

    cudaFree(camera);
    cudaFree(ans);

    printf("\n Camera test pass...\n");
    printf("===================\n");
}

__global__ 
void testRenderKernel(Camera *camera, ivec3 *colors, int width, int height){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= width || j >= height) return;
    float r = float(i)/width;
    float g = float(j)/width;
    float b = 0.2;

    int ir = int(255.99 * r);
    int ig = int(255.99 * g);
    int ib = int(255.99 * b);
    
    int color_idx = j * width + i;
    colors[color_idx] = ivec3(ir, ig, ib);
}

void testRender() {
    cudaError_t cuda_ret;
    printf("\n Test Render on GPU... \n");

    /* Camera parameters */
    vec3 pos    = vec3(0.0,0.0,0.0);
    vec3 look_at= vec3(0.0, 0.0, 1.0);
    vec3 up_vec = vec3(0.0, 1.0, 0.0);
    int width  = 640;
    int height = 480;
    Camera *camera;
    ivec3 *colors;
    
    /*Malloc memory and set up for camer */
    cuda_ret = cudaMallocManaged(&camera, sizeof(Camera));
    cuda_ret = cudaMallocManaged(&colors, sizeof(ivec3) * (width * height));
    if(cuda_ret != cudaSuccess){
        printf("Unable to malloc memory for parameters\n");
        cudaFree(camera);
        cudaFree(colors);
        return;
    }
    camera->Position_And_Aim_Camera(&pos, &look_at, &up_vec);
    camera->Focus_Camera(1.0, (double)width / height, 30.0 * (pi / 180));
    camera->Set_Resolution(width, height);

    int tx=8, ty = 8;

    dim3 blocks(width/tx + 1, height/ty +1);
    dim3 threads(tx, ty);
    testRenderKernel<<<blocks, threads>>>(camera, colors, width, height);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess){
        printf("Synchronize failed \n");
        cudaFree(camera);
        cudaFree(colors);
        return;
    }

   
    Dump_png(colors, width,height, "./result/test.png");
    cudaFree(camera);
    cudaFree(colors);

    printf("\n Render test pass, pls check resule/test.png\n");
    printf("===================\n");
}

__global__ 
void testSphereKernel(Object *sphere, ivec3 *color){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // if(idx == 0){
        vec3 rpos(0.0, 0.0, 0.0), rdir(0.0, 0.0, 1.0);
        Ray ray(&rpos, &rdir);
        double dist = sphere->Intersection(ray); 
        if(dist != __DBL_MAX__){
            (*color)[0] = int(100 * (*sphere).color[0]);
            (*color)[1] = int(100 * (*sphere).color[1]);
            (*color)[2] = int(100 * (*sphere).color[2]);
        }
    // }
    return;
}

void testSphere(){
    cudaError_t cuda_ret;
    printf("\n Test Sphere on GPU... \n");

    /* Camera parameters */
    Sphere *sphere;
    ivec3 *color;
    
    /*Malloc memory and set up for camer */
    cuda_ret = cudaMallocManaged(&sphere, sizeof(Sphere));
    cuda_ret = cudaMallocManaged(&color, sizeof(ivec3));
    if(cuda_ret != cudaSuccess){
        printf("Unable to malloc memory for parameters\n");
        cudaFree(sphere);
        cudaFree(color);
        return;
    }
    vec3 pos(0.0, 0.0, 0.0);
    double radius = 3.0;
    vec3 c(0.62, 0.41, 0.32);
    *sphere = Sphere(pos, radius, c);

    testSphereKernel<<<222, 33>>>(sphere, color);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess){
        printf("Synchronize failed \n");
        cudaFree(color);
        cudaFree(sphere);
        return;
    }

    assert(
            (*color)[0] == int(c[0]*100) &&
            (*color)[1] == int(c[1]*100) &&
            (*color)[2] == int(c[2]*100)
            );

    cudaFree(color);
    cudaFree(sphere);

    printf("\n Sphere test pass\n");
    printf("===================\n");
}
