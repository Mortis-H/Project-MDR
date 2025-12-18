# Generate Pass header files
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AMDISA)
add_public_tablegen_target(AMDISAPassIncGen)

