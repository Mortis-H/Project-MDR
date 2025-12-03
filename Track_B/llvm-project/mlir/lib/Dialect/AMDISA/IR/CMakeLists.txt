#===- AMDISA Dialect -----------------------------------------------------===#

add_mlir_dialect_library(MLIRAMDISA
  AMDISAOps.cpp

  DEPENDS
  MLIRAMDISAIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRBytecodeOpInterface
)
