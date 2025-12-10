/**
 * parse_utils.h
 * 
 * 提供可重用的 AMD GCN 組合語言解析函數
 * 從 main.cpp 提取出來，供其他程式使用
 */

#ifndef PARSE_UTILS_H
#define PARSE_UTILS_H

#include "AMDGCNAssembly.h"
#include <string>

/**
 * 解析 AMD GCN 組合語言檔案
 * 
 * @param filename 要解析的 .s 檔案路徑
 * @param triple Target triple (預設: "amdgcn-amd-amdhsa")
 * @param mcpu Target CPU (預設: "gfx950")
 * @return 包含所有解析資訊的 AMDGCNAssembly 對象
 */
AMDGCNAssembly parseAMDGCNAssembly(const std::string &filename,
                                    const std::string &triple = "amdgcn-amd-amdhsa",
                                    const std::string &mcpu = "gfx950");

#endif // PARSE_UTILS_H

