#ifndef AMDGPU_METADATA_H
#define AMDGPU_METADATA_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>
#include <map>

// ============================================================================
// AMD GPU Metadata 數據結構
// ============================================================================

/// Kernel 參數描述
struct KernelArg {
  // 使用 vector 保持屬性的插入順序（YAML 中的順序很重要）
  std::vector<std::pair<std::string, std::string>> properties;
  
  // 便捷訪問方法（向後兼容）
  int offset() const {
    for (const auto &prop : properties) {
      if (prop.first == "offset") {
        char *endptr = nullptr;
        long val = std::strtol(prop.second.c_str(), &endptr, 10);
        if (endptr != prop.second.c_str()) {
          return static_cast<int>(val);
        }
      }
    }
    return 0;
  }
  
  int size() const {
    for (const auto &prop : properties) {
      if (prop.first == "size") {
        char *endptr = nullptr;
        long val = std::strtol(prop.second.c_str(), &endptr, 10);
        if (endptr != prop.second.c_str()) {
          return static_cast<int>(val);
        }
      }
    }
    return 0;
  }
  
  std::string valueKind() const {
    for (const auto &prop : properties) {
      if (prop.first == "value_kind") {
        return prop.second;
      }
    }
    return "";
  }
  
  std::string name() const {
    for (const auto &prop : properties) {
      if (prop.first == "name") {
        return prop.second;
      }
    }
    return "";
  }
  
  std::string typeName() const {
    for (const auto &prop : properties) {
      if (prop.first == "type_name") {
        return prop.second;
      }
    }
    return "";
  }
  
  std::string addressSpace() const {
    for (const auto &prop : properties) {
      if (prop.first == "address_space") {
        return prop.second;
      }
    }
    return "";
  }
  
  // 通用屬性訪問
  std::string getProperty(const std::string &key) const {
    for (const auto &prop : properties) {
      if (prop.first == key) {
        return prop.second;
      }
    }
    return "";
  }
  
  bool hasProperty(const std::string &key) const {
    for (const auto &prop : properties) {
      if (prop.first == key) {
        return true;
      }
    }
    return false;
  }
  
  // 取得所有屬性（按插入順序）
  const std::vector<std::pair<std::string, std::string>>& getAllProperties() const {
    return properties;
  }
  
  KernelArg() = default;
  
  void print(llvm::raw_ostream &OS, int indent = 0) const {
    std::string ind(indent, ' ');
    OS << ind << "- Arg:\n";
    
    // 按插入順序輸出所有屬性（保持 YAML 原始順序）
    for (const auto &prop : properties) {
      OS << ind << "    " << prop.first << ": " << prop.second << "\n";
    }
  }
};

/// Kernel 資訊
struct KernelInfo {
  std::string name;
  std::string symbol;
  
  // 資源使用
  int agprCount;
  int sgprCount;
  int vgprCount;
  int sgprSpillCount;
  int vgprSpillCount;
  
  // 記憶體配置
  int groupSegmentFixedSize;
  int privateSegmentFixedSize;
  int kernargSegmentSize;
  int kernargSegmentAlign;
  
  // 工作群組配置
  int maxFlatWorkgroupSize;
  int wavefrontSize;
  bool uniformWorkGroupSize;
  bool usesDynamicStack;
  
  // 語言資訊
  std::string language;
  std::vector<int> languageVersion;
  
  // 參數列表
  std::vector<KernelArg> args;
  
  // 額外屬性 (用於存儲未明確定義的欄位)
  std::map<std::string, std::string> extraProperties;
  
  KernelInfo() 
    : agprCount(0), sgprCount(0), vgprCount(0),
      sgprSpillCount(0), vgprSpillCount(0),
      groupSegmentFixedSize(0), privateSegmentFixedSize(0),
      kernargSegmentSize(0), kernargSegmentAlign(0),
      maxFlatWorkgroupSize(0), wavefrontSize(64),
      uniformWorkGroupSize(false), usesDynamicStack(false) {}
  
  void print(llvm::raw_ostream &OS) const {
    OS << "Kernel: " << name << "\n";
    OS << "  Symbol: " << symbol << "\n";
    OS << "  Resource Usage:\n";
    OS << "    AGPR Count: " << agprCount << "\n";
    OS << "    SGPR Count: " << sgprCount << "\n";
    OS << "    VGPR Count: " << vgprCount << "\n";
    OS << "    SGPR Spills: " << sgprSpillCount << "\n";
    OS << "    VGPR Spills: " << vgprSpillCount << "\n";
    OS << "  Memory Configuration:\n";
    OS << "    Group Segment: " << groupSegmentFixedSize << " bytes\n";
    OS << "    Private Segment: " << privateSegmentFixedSize << " bytes\n";
    OS << "    Kernarg Segment: " << kernargSegmentSize << " bytes (align: " 
       << kernargSegmentAlign << ")\n";
    OS << "  Workgroup Configuration:\n";
    OS << "    Max Flat Workgroup Size: " << maxFlatWorkgroupSize << "\n";
    OS << "    Wavefront Size: " << wavefrontSize << "\n";
    OS << "    Uniform Workgroup: " << (uniformWorkGroupSize ? "yes" : "no") << "\n";
    OS << "  Language: " << language;
    if (!languageVersion.empty()) {
      OS << " ";
      for (size_t i = 0; i < languageVersion.size(); ++i) {
        OS << languageVersion[i];
        if (i < languageVersion.size() - 1) OS << ".";
      }
    }
    OS << "\n";
    
    if (!args.empty()) {
      OS << "  Arguments (" << args.size() << "):\n";
      for (const auto &arg : args) {
        arg.print(OS, 4);
      }
    }
    
    if (!extraProperties.empty()) {
      OS << "  Extra Properties:\n";
      for (const auto &kv : extraProperties) {
        OS << "    " << kv.first << ": " << kv.second << "\n";
      }
    }
  }
};

/// AMD GPU Metadata 容器
class AMDGPUMetadata {
public:
  // Kernels 列表
  std::vector<KernelInfo> kernels;
  
  // Target 資訊
  std::string target;  // e.g., "amdgcn-amd-amdhsa--gfx950"
  
  // Version 資訊
  std::vector<int> version;  // e.g., [1, 2]
  
  // 原始 YAML 文本（保留以供參考或重新輸出）
  std::string rawYAML;
  
  // Metadata 在文件中的行範圍
  size_t startLine;
  size_t endLine;
  
  AMDGPUMetadata() : startLine(0), endLine(0) {}
  
  /// 添加 kernel 資訊
  void addKernel(const KernelInfo &kernel) {
    kernels.push_back(kernel);
  }
  
  /// 根據名稱查找 kernel
  const KernelInfo* findKernel(llvm::StringRef name) const {
    for (const auto &k : kernels) {
      if (k.name == name.str()) return &k;
    }
    return nullptr;
  }
  
  /// 取得 kernel 數量
  size_t getKernelCount() const { return kernels.size(); }
  
  /// 檢查是否有 metadata
  bool isEmpty() const { return kernels.empty() && rawYAML.empty(); }
  
  /// 打印統計摘要
  void printSummary(llvm::raw_ostream &OS) const {
    OS << "=== AMD GPU Metadata Summary ===\n";
    OS << "Target: " << target << "\n";
    OS << "Version: ";
    for (size_t i = 0; i < version.size(); ++i) {
      OS << version[i];
      if (i < version.size() - 1) OS << ".";
    }
    OS << "\n";
    OS << "Kernels: " << kernels.size() << "\n";
    OS << "Lines: " << startLine << "-" << endLine << "\n\n";
    
    for (const auto &kernel : kernels) {
      kernel.print(OS);
      OS << "\n";
    }
  }
  
  /// 打印原始 YAML
  void printRawYAML(llvm::raw_ostream &OS) const {
    if (!rawYAML.empty()) {
      OS << rawYAML;
    }
  }
  
  /// 檢查特定行是否在 metadata 範圍內
  bool containsLine(size_t lineNum) const {
    return lineNum >= startLine && lineNum <= endLine;
  }
};

// ============================================================================
// Metadata 解析器
// ============================================================================

/// 簡易 YAML 解析器（針對 AMD GPU metadata 格式）
class MetadataParser {
public:
  /// 解析 metadata YAML 文本
  static bool parse(const std::string &yamlText, 
                    size_t startLine, 
                    size_t endLine,
                    AMDGPUMetadata &outMetadata);
  
private:
  /// 解析 kernel 區塊
  static bool parseKernel(const std::vector<std::string> &lines,
                          size_t &idx,
                          KernelInfo &outKernel);
  
  /// 解析 kernel 參數
  static bool parseKernelArg(const std::vector<std::string> &lines,
                             size_t &idx,
                             KernelArg &outArg);
  
  /// 取得縮排層級
  static int getIndentLevel(const std::string &line);
  
  /// 移除前導空白並返回 key: value
  static bool parseKeyValue(const std::string &line, 
                           std::string &key, 
                           std::string &value);
  
  /// 解析整數
  static bool parseInt(const std::string &str, int &outValue);
  
  /// 解析布林值
  static bool parseBool(const std::string &str, bool &outValue);
};

#endif // AMDGPU_METADATA_H

