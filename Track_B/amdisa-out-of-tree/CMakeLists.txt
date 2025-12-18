cmake_minimum_required(VERSION 3.20.0)
project(amdisa-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# ============================================================================
# Out-of-tree build configuration
# ============================================================================
# This project is designed to be built as a standalone out-of-tree project
# against an installed LLVM/MLIR distribution.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Out-of-tree build
  find_package(MLIR REQUIRED CONFIG)
  
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  # Define the default arguments to use when testing with lit.
  set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # In-tree build (fallback, not recommended for this project)
  message(WARNING "In-tree build detected. This project is designed for out-of-tree builds.")
endif()

# ============================================================================
# Project directories
# ============================================================================
set(AMDISA_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(AMDISA_BINARY_DIR ${PROJECT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${AMDISA_SOURCE_DIR}/include)
include_directories(${AMDISA_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# ============================================================================
# Build subdirectories
# ============================================================================
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

