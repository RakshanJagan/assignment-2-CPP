cmake_minimum_required(VERSION 3.10)

# Project name
project(ASSIGNMENT_2_TEST)

include(FetchContent)

FetchContent_Declare(
  json 
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Source files
set(SOURCES 
    src/main.cpp
    utilities/src/convolution.cpp 
    utilities/src/dense.cpp 
    utilities/src/pooling.cpp
)

# Add executable
add_executable(ASSIGNMENT_2_TEST ${SOURCES})
target_link_libraries(ASSIGNMENT_2_TEST PRIVATE nlohmann_json::nlohmann_json)

# Include directories
target_include_directories(ASSIGNMENT_2_TEST PRIVATE utilities/include)

# Specify output directory (optional)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)  
