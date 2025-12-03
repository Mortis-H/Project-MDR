set(LLVM_TARGET_DEFINITIONS AMDISAOps.td)

# 產生 OP 宣告（header）
mlir_tablegen(AMDISAOps.h.inc -gen-op-decls -dialect=amdisa)

# 產生 OP 定義（impl）
mlir_tablegen(AMDISAOps.cpp.inc -gen-op-defs -dialect=amdisa)

# 產生 dialect 宣告
mlir_tablegen(AMDISAOpsDialect.h.inc -gen-dialect-decls -dialect=amdisa)

# 產生 dialect 定義
mlir_tablegen(AMDISAOpsDialect.cpp.inc -gen-dialect-defs -dialect=amdisa)

add_public_tablegen_target(MLIRAMDISAIncGen)

# 可選：生成 dialect 文檔
# add_mlir_doc(AMDISAOps AMDISA Dialects/ -gen-dialect-doc)
