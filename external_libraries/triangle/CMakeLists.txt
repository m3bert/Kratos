set(CMAKE_INCLUDE_CURRENT_DIR ON)

# generate variables with the sources
set(TRIANGLE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/triangle.c)

# removing optimizations
SET_SOURCE_FILES_PROPERTIES(${TRIANGLE_SOURCES} PROPERTIES COMPILE_FLAGS -O0)

add_definitions( -DTRILIBRARY )
add_definitions( -w )

# perform platform specific defintions
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  add_definitions(-DCPU86)
  add_definitions(-DNO_TIMER)
else(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  add_definitions(-DLINUX)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

###############################################################

add_library(triangle STATIC ${TRIANGLE_SOURCES})

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  target_link_libraries(triangle m)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

set(TRIANGLE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TRIANGLE_LIBRARIES triangle)

message(STATUS "triangle : ${TRIANGLE_INCLUDE_DIR}" )
# install(TARGETS triangle DESTINATION libs )
