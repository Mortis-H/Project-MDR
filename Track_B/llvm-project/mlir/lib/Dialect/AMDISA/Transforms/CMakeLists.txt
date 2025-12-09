add_mlir_dialect_library(MLIRAMDISATransforms
  LowerToGPUInlineAsm.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AMDISA

  DEPENDS
    MLIRAMDISAPassIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRGPUDialect
    MLIRLLVMDialect
    MLIRPass
    MLIRSupport
)
