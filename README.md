# CS217-RayTracing

Project for CS217 GPU architecture. This branch is GPU Version.

We implemented a basic ray tracing program, which include basic objects and light sources. It also support phong shader. Running much faster than the CPU Version.

## How to run

Enter the root of this project, using following commands to compile:
```bash
make clean
make
```

After compile, you will have three executables **scene1**, **scene2** and **scene3**. 

Using this command to run the programs:
```bash
./scene1
./scene2
./scene3
```

Then you can see the result images in result fold.


## Reference 
https://in1weekend.blogspot.com/2016/01/ray-tracing-in-one-weekend.html

https://news.developer.nvidia.com/ray-tracing-essentials-part-1-basics-of-ray-tracing/

https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/
