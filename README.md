# CS217-RayTracing

Project for CS217 GPU architecture. This branch is CPU Version.

We implemented a basic ray tracing program, which include basic objects and light sources. It also support phong shader. Used as comparsion with GPU Version.

## How to run
Please make sure [CMake](https://cmake.org/) is installed on your machine. 

Enter the root of this project, using following commands to compile:
```bash
./start.sh
```

After compile, you will have three executables in **build** fold. Using this command to 
run the programs:
```bash
./build/scene1
./build/scene2
./build/scene3
```

Then you can see the result images in result fold.

## Reference 
https://in1weekend.blogspot.com/2016/01/ray-tracing-in-one-weekend.html

https://news.developer.nvidia.com/ray-tracing-essentials-part-1-basics-of-ray-tracing/

https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/
