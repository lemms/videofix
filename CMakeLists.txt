project (VideoProcessor)
cmake_minimum_required (VERSION 2.6)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set (LIBRARY_OUTPUT_PATH    ${PROJECT_BINARY_DIR}/lib)

set (Boost_USE_STATIC_LIBS    ON)
set (Boost_USE_MULTITHREADED  ON)
set (Boost_USE_STATIC_RUNTIME OFF)
set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
find_package (Boost COMPONENTS system filesystem program_options REQUIRED)
find_package (OpenCV 320 REQUIRED)
find_package (OpenMP REQUIRED)
if (OpenMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OpenMP_FOUND)

add_subdirectory (src)
