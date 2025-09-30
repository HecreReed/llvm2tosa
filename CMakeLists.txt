cmake_minimum_required(VERSION 3.13.4)
project(llvm2tosa)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Create the main converter executable
add_executable(llvm2tosa 
    main.cpp
    src/CompleteConverter.cpp
)

# Create demo executable if simple_converter_demo.cpp exists
if(EXISTS "${CMAKE_SOURCE_DIR}/examples/simple_converter_demo.cpp")
    add_executable(converter_demo 
        examples/simple_converter_demo.cpp
        src/CompleteConverter.cpp
    )
endif()

# Create unit test executable if unit_test.cpp exists
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/unit_test.cpp")
    add_executable(unit_tests
        tests/unit_test.cpp
        src/CompleteConverter.cpp
    )
endif()

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
    # target_link_libraries(llvm2tosa PRIVATE ${MLIR_LIBRARIES})
else()
    message(STATUS "LLVM/MLIR not found - building standalone converter")
endif()

# Install targets
install(TARGETS llvm2tosa DESTINATION bin)
install(DIRECTORY examples/ DESTINATION share/llvm2tosa/examples)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests")
    install(DIRECTORY tests/ DESTINATION share/llvm2tosa/tests)
endif()