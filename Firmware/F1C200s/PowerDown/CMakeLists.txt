cmake_minimum_required(VERSION 2.8)

project(PowerDown)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -static-libstdc++")

add_executable(${PROJECT_NAME} main.cpp
    ../base/mutex.cpp
    ../base/i2cbase.cpp
    )
target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}
    PUBLIC ${PROJECT_SOURCE_DIR}/../base
    )
