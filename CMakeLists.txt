cmake_minimum_required(VERSION 3.0.2)
project(mpc_ros)

find_package(catkin REQUIRED)

# Eigen #
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
link_directories(${EIGEN3_LIBRARY_DIRS})

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES first_order_mpc
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(example1 src/example1.cpp)
target_link_libraries(example1 ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} qpOASES)

add_executable(first_order_mpc src/first_order_mpc.cpp)
target_link_libraries(first_order_mpc ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} qpOASES)

add_executable(second_order_mpc src/second_order_mpc.cpp)
target_link_libraries(second_order_mpc ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} qpOASES)
