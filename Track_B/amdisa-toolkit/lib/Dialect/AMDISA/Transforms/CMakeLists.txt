add_mlir_dialect_library(MLIRAMDISATransforms
  LowerToGPUInlineAsm.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  DEPENDS
  MLIRAMDISAPassIncGen
  MLIRAMDISAIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAMDISA
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRSupport
)
