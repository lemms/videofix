message("Add classify executable")
add_executable(classify src/classify.cpp)
message("Including: ${CMAKE_SOURCE_DIR}/src/classifiers/include ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}")
include_directories(${CMAKE_SOURCE_DIR}/src/classifiers/include ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
message("Linking: ${OpenCV_LIBRARIES} ${Boost_LIBRARIES}")
target_link_libraries(classify classifiers ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
