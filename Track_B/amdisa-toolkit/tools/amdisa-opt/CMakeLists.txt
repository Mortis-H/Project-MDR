#===- CMakeLists.txt - AMDISA opt tool ----------------------------------===#

# Collect all registered dialect libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# Collect all registered conversion libraries
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# Collect all registered extension libraries
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# --------------------------------------------------------------------------
# Tool: amdisa-opt
# --------------------------------------------------------------------------

add_mlir_tool(amdisa-opt
  amdisa-opt.cpp

  DEPENDS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRAMDISAIncGen
  MLIRAMDISAPassIncGen
)

llvm_update_compile_flags(amdisa-opt)

# Link all required libraries
target_link_libraries(amdisa-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  
  # AMDISA libraries
  MLIRAMDISA
  MLIRAMDISATransforms
  
  # Register all MLIR dialects and passes (required for opt tool)
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  
  # MLIR core libraries
  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
)

mlir_check_link_libraries(amdisa-opt)

