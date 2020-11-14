cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
Project(RayTracing)

# include path
include_directories("include")

# set libray
file(GLOB LIB_SOURCES src/*.cpp)
file(GLOB LIB_HEADERS include/*.h)
add_library(RTLIB ${LIB_SOURCES} ${LIB_HEADERS})

add_executable(RayTracing main/main.cpp)
target_link_libraries(RayTracing RTLIB)

add_executable(TestRT test/test.cpp)
target_link_libraries(RayTracing RTLIB)

set(CMAKE_CXX_STANDARD_REQUIRED YES)
