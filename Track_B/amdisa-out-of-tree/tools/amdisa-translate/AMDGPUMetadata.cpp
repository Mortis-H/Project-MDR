#include "AMDGPUMetadata.h"
#include <sstream>
#include <cctype>
#include <algorithm>

// ============================================================================
// 輔助函數
// ============================================================================

int MetadataParser::getIndentLevel(const std::string &line) {
  int indent = 0;
  for (char c : line) {
    if (c == ' ') indent++;
    else if (c == '\t') indent += 4;
    else break;
  }
  return indent;
}

bool MetadataParser::parseKeyValue(const std::string &line, 
                                    std::string &key, 
                                    std::string &value) {
  size_t colonPos = line.find(':');
  if (colonPos == std::string::npos) return false;
  
  // 提取 key
  std::string rawKey = line.substr(0, colonPos);
  // 移除前導空白和 '.' 或 '-'
  size_t keyStart = rawKey.find_first_not_of(" \t.-");
  if (keyStart == std::string::npos) return false;
  key = rawKey.substr(keyStart);
  
  // 提取 value
  if (colonPos + 1 < line.size()) {
    std::string rawValue = line.substr(colonPos + 1);
    size_t valueStart = rawValue.find_first_not_of(" \t");
    if (valueStart != std::string::npos) {
      value = rawValue.substr(valueStart);
      // 移除尾隨空白
      size_t valueEnd = value.find_last_not_of(" \t\r\n");
      if (valueEnd != std::string::npos) {
        value = value.substr(0, valueEnd + 1);
      }
    } else {
      value = "";
    }
  } else {
    value = "";
  }
  
  return true;
}

bool MetadataParser::parseInt(const std::string &str, int &outValue) {
  if (str.empty()) return false;
  
  // 手動解析整數（避免使用異常）
  size_t i = 0;
  bool negative = false;
  
  if (str[0] == '-') {
    negative = true;
    i = 1;
  } else if (str[0] == '+') {
    i = 1;
  }
  
  if (i >= str.size() || !std::isdigit(str[i])) {
    return false;
  }
  
  int result = 0;
  while (i < str.size() && std::isdigit(str[i])) {
    result = result * 10 + (str[i] - '0');
    i++;
  }
  
  // 確保所有字符都被解析
  if (i != str.size()) {
    return false;
  }
  
  outValue = negative ? -result : result;
  return true;
}

bool MetadataParser::parseBool(const std::string &str, bool &outValue) {
  std::string lower = str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  
  if (lower == "true" || lower == "1" || lower == "yes") {
    outValue = true;
    return true;
  } else if (lower == "false" || lower == "0" || lower == "no") {
    outValue = false;
    return true;
  }
  return false;
}

// ============================================================================
// Kernel 參數解析
// ============================================================================

bool MetadataParser::parseKernelArg(const std::vector<std::string> &lines,
                                     size_t &idx,
                                     KernelArg &outArg) {
  int baseIndent = getIndentLevel(lines[idx]);
  
  // 處理第一行，可能是 "- " 單獨一行，或 "- .field: value" 在同一行
  std::string firstLine = lines[idx];
  size_t dashPos = firstLine.find('-');
  if (dashPos != std::string::npos) {
    // 找到 '-' 後面的內容
    size_t afterDash = dashPos + 1;
    size_t nextNonSpace = firstLine.find_first_not_of(" \t", afterDash);
    if (nextNonSpace != std::string::npos && firstLine[nextNonSpace] == '.') {
      // "- .field: value" 格式，解析這個欄位
      std::string fieldPart = firstLine.substr(nextNonSpace);
      std::string key, value;
      if (parseKeyValue(fieldPart, key, value)) {
        // 去掉前綴的點
        std::string cleanKey = key;
        if (!cleanKey.empty() && cleanKey[0] == '.') {
          cleanKey = cleanKey.substr(1);
        }
        outArg.properties.push_back({cleanKey, value});
      }
    }
  }
  
  idx++; // 移到下一行
  
  while (idx < lines.size()) {
    const std::string &line = lines[idx];
    
    // 空行或註解
    if (line.empty() || line.find_first_not_of(" \t") == std::string::npos) {
      idx++;
      continue;
    }
    
    int indent = getIndentLevel(line);
    
    // 如果縮排回到同級或更小，表示這個 arg 結束
    if (indent <= baseIndent) break;
    
    std::string key, value;
    if (!parseKeyValue(line, key, value)) {
      idx++;
      continue;
    }
    
    // 動態存儲所有屬性（去掉前綴的點）
    std::string cleanKey = key;
    if (!cleanKey.empty() && cleanKey[0] == '.') {
      cleanKey = cleanKey.substr(1);
    }
    
    outArg.properties.push_back({cleanKey, value});
    
    idx++;
  }
  
  return true;
}

// ============================================================================
// Kernel 解析
// ============================================================================

bool MetadataParser::parseKernel(const std::vector<std::string> &lines,
                                  size_t &idx,
                                  KernelInfo &outKernel) {
  int baseIndent = getIndentLevel(lines[idx]);
  
  // 處理第一行，可能是 "- " 單獨一行，或 "- .field: value" 在同一行
  std::string firstLine = lines[idx];
  size_t firstNonSpace = firstLine.find_first_not_of(" \t");
  if (firstNonSpace != std::string::npos && 
      firstLine[firstNonSpace] == '-' &&
      firstNonSpace + 1 < firstLine.size()) {
    // 檢查 '-' 後面是否有內容
    size_t afterDash = firstNonSpace + 1;
    size_t nextNonSpace = firstLine.find_first_not_of(" \t", afterDash);
    if (nextNonSpace != std::string::npos) {
      // "- .field: value" 格式，解析這個欄位
      std::string fieldPart = firstLine.substr(nextNonSpace);
      std::string key, value;
      if (parseKeyValue(fieldPart, key, value)) {
        // 處理第一個欄位（通常是 .agpr_count）
        if (key == "agpr_count" || key == ".agpr_count") {
          parseInt(value, outKernel.agprCount);
        }
      }
    }
  }
  
  idx++; // 移到下一行
  
  while (idx < lines.size()) {
    const std::string &line = lines[idx];
    
    // 空行或註解
    if (line.empty() || line.find_first_not_of(" \t") == std::string::npos) {
      idx++;
      continue;
    }
    
    int indent = getIndentLevel(line);
    
    // 如果縮排回到同級或更小，表示這個 kernel 結束
    if (indent <= baseIndent) break;
    
    std::string key, value;
    if (!parseKeyValue(line, key, value)) {
      idx++;
      continue;
    }
    
    // 解析各個欄位（處理帶點和不帶點的鍵名）
    if (key == "name" || key == ".name") {
      outKernel.name = value;
    } else if (key == "symbol" || key == ".symbol") {
      outKernel.symbol = value;
    } else if (key == "agpr_count" || key == ".agpr_count") {
      parseInt(value, outKernel.agprCount);
    } else if (key == "sgpr_count" || key == ".sgpr_count") {
      parseInt(value, outKernel.sgprCount);
    } else if (key == "vgpr_count" || key == ".vgpr_count") {
      parseInt(value, outKernel.vgprCount);
    } else if (key == "sgpr_spill_count" || key == ".sgpr_spill_count") {
      parseInt(value, outKernel.sgprSpillCount);
    } else if (key == "vgpr_spill_count" || key == ".vgpr_spill_count") {
      parseInt(value, outKernel.vgprSpillCount);
    } else if (key == "group_segment_fixed_size" || key == ".group_segment_fixed_size") {
      parseInt(value, outKernel.groupSegmentFixedSize);
    } else if (key == "private_segment_fixed_size" || key == ".private_segment_fixed_size") {
      parseInt(value, outKernel.privateSegmentFixedSize);
    } else if (key == "kernarg_segment_size" || key == ".kernarg_segment_size") {
      parseInt(value, outKernel.kernargSegmentSize);
    } else if (key == "kernarg_segment_align" || key == ".kernarg_segment_align") {
      parseInt(value, outKernel.kernargSegmentAlign);
    } else if (key == "max_flat_workgroup_size" || key == ".max_flat_workgroup_size") {
      parseInt(value, outKernel.maxFlatWorkgroupSize);
    } else if (key == "wavefront_size" || key == ".wavefront_size") {
      parseInt(value, outKernel.wavefrontSize);
    } else if (key == "uniform_work_group_size" || key == ".uniform_work_group_size") {
      parseBool(value, outKernel.uniformWorkGroupSize);
    } else if (key == "uses_dynamic_stack" || key == ".uses_dynamic_stack") {
      parseBool(value, outKernel.usesDynamicStack);
    } else if (key == "language" || key == ".language") {
      outKernel.language = value;
    } else if (key == "language_version" || key == ".language_version") {
      // 下一行應該是版本號列表
      idx++;
      while (idx < lines.size()) {
        const std::string &versionLine = lines[idx];
        int versionIndent = getIndentLevel(versionLine);
        if (versionIndent <= indent) break;
        
        std::string trimmed = versionLine;
        trimmed.erase(0, trimmed.find_first_not_of(" \t-"));
        int ver;
        if (parseInt(trimmed, ver)) {
          outKernel.languageVersion.push_back(ver);
        }
        idx++;
      }
      continue; // 已經 idx++，所以這裡 continue
    } else if (key == "args" || key == ".args") {
      // 下一行開始是參數列表
      idx++;
      while (idx < lines.size()) {
        const std::string &argLine = lines[idx];
        int argIndent = getIndentLevel(argLine);
        if (argIndent <= indent) break;
        
        // 檢查是否是 "- " 開頭（新參數）
        std::string trimmed = argLine;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));
        if (trimmed.empty() || trimmed[0] != '-') {
          idx++;
          continue;
        }
        
        KernelArg arg;
        parseKernelArg(lines, idx, arg);
        outKernel.args.push_back(arg);
      }
      continue; // 已經 idx++，所以這裡 continue
    } else {
      // 其他未知欄位存入 extraProperties
      if (!value.empty()) {
        outKernel.extraProperties[key] = value;
      }
    }
    
    idx++;
  }
  
  return true;
}

// ============================================================================
// 主解析函數
// ============================================================================

bool MetadataParser::parse(const std::string &yamlText, 
                            size_t startLine, 
                            size_t endLine,
                            AMDGPUMetadata &outMetadata) {
  outMetadata.rawYAML = yamlText;
  outMetadata.startLine = startLine;
  outMetadata.endLine = endLine;
  
  // 分割成行
  std::vector<std::string> lines;
  std::stringstream ss(yamlText);
  std::string line;
  while (std::getline(ss, line)) {
    lines.push_back(line);
  }
  
  // 逐行解析
  for (size_t i = 0; i < lines.size(); ++i) {
    const std::string &currentLine = lines[i];
    
    // 跳過空行和 YAML 開始/結束標記
    if (currentLine.empty() || 
        currentLine.find("---") != std::string::npos ||
        currentLine.find("...") != std::string::npos) {
      continue;
    }
    
    std::string key, value;
    if (!parseKeyValue(currentLine, key, value)) continue;
    
    // 解析頂層欄位
    if (key == "amdhsa.kernels") {
      // 下一行開始是 kernel 列表
      i++;
      while (i < lines.size()) {
        const std::string &kernelLine = lines[i];
        int indent = getIndentLevel(kernelLine);
        
        // 檢查是否是 "- " 開頭（新 kernel）
        std::string trimmed = kernelLine;
        size_t firstNonSpace = trimmed.find_first_not_of(" \t");
        if (firstNonSpace == std::string::npos) {
          i++;
          continue;
        }
        trimmed = trimmed.substr(firstNonSpace);
        
        // 如果不是 '-' 開頭且縮排為 0，表示離開了 kernels 區塊
        if (indent == 0 && trimmed[0] != '-') {
          i--; // 回退一行，讓外層循環可以處理這一行
          break;
        }
        
        if (trimmed[0] == '-') {
          KernelInfo kernel;
          parseKernel(lines, i, kernel);
          outMetadata.kernels.push_back(kernel);
        } else {
          i++;
        }
      }
    } else if (key == "amdhsa.target") {
      outMetadata.target = value;
    } else if (key == "amdhsa.version") {
      // 下一行應該是版本號列表
      i++;
      while (i < lines.size()) {
        const std::string &versionLine = lines[i];
        int indent = getIndentLevel(versionLine);
        if (indent == 0) break;
        
        std::string trimmed = versionLine;
        trimmed.erase(0, trimmed.find_first_not_of(" \t-"));
        int ver;
        if (parseInt(trimmed, ver)) {
          outMetadata.version.push_back(ver);
        }
        i++;
      }
    }
  }
  
  return !outMetadata.isEmpty();
}

