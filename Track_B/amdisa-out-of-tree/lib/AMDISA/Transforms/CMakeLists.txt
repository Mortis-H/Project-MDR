add_mlir_library(MLIRAMDISATransforms
  LowerToGPUInlineAsm.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/AMDISA

  DEPENDS
  AMDISAPassIncGen

  LINK_LIBS PUBLIC
  MLIRAMDISA
  MLIRIR
  MLIRPass
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRTransformUtils
)

