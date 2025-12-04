#===- CMakeLists.txt - AMDISA translate tool -----------------------------===#

set(LLVM_LINK_COMPONENTS
  Support
)

# Collect all registered dialect libs
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# Collect all registered translation libs (optional)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# --------------------------------------------------------------------------
# Tool: amdisa-translate
# --------------------------------------------------------------------------

add_mlir_tool(amdisa-translate
  mdr.cpp
)

llvm_update_compile_flags(amdisa-translate)

mlir_target_link_libraries(amdisa-translate
  PRIVATE
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  )

target_link_libraries(amdisa-translate
  PRIVATE
  ${translation_libs}
  MLIRAMDISA
)

mlir_check_link_libraries(amdisa-translate)
