set(LLVM_LINK_COMPONENTS
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
)

# Collect all registered dialect libs
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# Collect all registered translation libs (optional)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# --------------------------------------------------------------------------
# Tool: amdisa-translate
# --------------------------------------------------------------------------

add_llvm_executable(amdisa-translate
  amdisa-translate.cpp
  AMDISAAsmParser.cpp
  AMDGCNAssembly.cpp
  AMDGPUMetadata.cpp
  parse_utils.cpp
)

llvm_update_compile_flags(amdisa-translate)

target_link_libraries(amdisa-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRAMDISA
  MLIRAMDISATransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRGPUDialect
  MLIRLLVMDialect
)

mlir_check_link_libraries(amdisa-translate)

