cmake_minimum_required(VERSION 3.20)
project(amdisa_annotate LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 設置 Release 模式以減少執行檔大小
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 你的 LLVM build 位置（請根據實際路徑修改）
set(LLVM_DIR "/home/morhuang/llvm-project/build/lib/cmake/llvm")

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_definitions(${LLVM_DEFINITIONS})

# 讓 CMake 找到 AddLLVM.cmake / HandleLLVMOptions.cmake
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(HandleLLVMOptions)

# 我們需要的 LLVM components
llvm_map_components_to_libnames(LLVM_LIBS
  Support
  Option
  Target
  TargetParser
  MC
  MCParser

  AMDGPUInfo
  AMDGPUDesc
  AMDGPUAsmParser
  AMDGPUCodeGen

  X86Info
  X86Desc
  X86AsmParser
  X86CodeGen
)

# 主程式：amdisa_annotate
add_executable(amdisa_annotate
  src/main.cpp
  src/AMDGPUMetadata.cpp
  src/AMDGCNAssembly.cpp
)

llvm_update_compile_flags(amdisa_annotate)

target_link_libraries(amdisa_annotate
  PRIVATE
    ${LLVM_LIBS}
)

# 自動 strip debug symbols 以減少執行檔大小
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_custom_command(TARGET amdisa_annotate POST_BUILD
    COMMAND ${CMAKE_STRIP} $<TARGET_FILE:amdisa_annotate>
    COMMENT "Stripping debug symbols from amdisa_annotate"
  )
endif()

# API 使用示範程式
add_executable(api_usage_demo
  examples/api_usage_demo.cpp
  src/parse_utils.cpp
  src/AMDGPUMetadata.cpp
  src/AMDGCNAssembly.cpp
)

llvm_update_compile_flags(api_usage_demo)

target_link_libraries(api_usage_demo
  PRIVATE
    ${LLVM_LIBS}
)

# 自動 strip debug symbols 以減少執行檔大小
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_custom_command(TARGET api_usage_demo POST_BUILD
    COMMAND ${CMAKE_STRIP} $<TARGET_FILE:api_usage_demo>
    COMMENT "Stripping debug symbols from api_usage_demo"
  )
endif()

# 安裝規則（可選）
install(TARGETS amdisa_annotate api_usage_demo
  RUNTIME DESTINATION bin
)
