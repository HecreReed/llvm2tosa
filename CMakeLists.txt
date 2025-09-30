cmake_minimum_required(VERSION 3.13.4)
project(llvm2tosa)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Create the converter executable
add_executable(converter converter.cpp)

# Optional: Add LLVM/MLIR support if available
find_package(LLVM QUIET)
find_package(MLIR QUIET)

if(LLVM_FOUND AND MLIR_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Found MLIR")
    
    include_directories(${LLVM_INCLUDE_DIRS})
    include_directories(${MLIR_INCLUDE_DIRS})
    
    separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
    add_definitions(${LLVM_DEFINITIONS_LIST})
    
    # Link MLIR libraries if building with full MLIR support
    # target_link_libraries(converter PRIVATE ${MLIR_LIBRARIES})
else()
    message(STATUS "LLVM/MLIR not found - building standalone converter")
endif()

# Install targets
install(TARGETS converter DESTINATION bin)
install(FILES examples.ll expected_tosa.mlir DESTINATION share/llvm2tosa)
install(PROGRAMS test.sh DESTINATION share/llvm2tosa)