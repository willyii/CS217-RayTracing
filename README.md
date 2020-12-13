# CS217-RayTracing
Project for CS217 GPU architecture. A basic cuda version raytracing program

## How to run

Enter the root of this project, using following commands to compile:
```bash
make clean
make
```

After compile, you will have one executable **rayTracing**. Using this command to 
run the test program:
```bash
./rayTracing
```

This program will add new file named "test.png" in **result** fold.

## TODO
- Object and sphere class that can be used in GPU and CPU
- Shader function
- Parse function 

## Reference 
https://in1weekend.blogspot.com/2016/01/ray-tracing-in-one-weekend.html

https://news.developer.nvidia.com/ray-tracing-essentials-part-1-basics-of-ray-tracing/

https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/
