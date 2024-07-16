/* ====================================================
#   Copyright (C) 2024 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : path.h
#   Last Modified : 2024-07-16 10:47
#   Describe      : 
#
# ====================================================*/


#pragma once

#include <string>
#include <vector>


namespace tofu {
namespace io {


void SplitStringPath(const std::string& full_path,
                     std::string* dir_part,
                     std::string* file_part);

std::string ConcatStringPath(const std::vector<std::string>& sections,
                             const std::string& delim = ".");

std::string GetPathPrefix(const std::string& full_path,
                          const std::string& delim = "/");

bool CreateDirWithRetry(const std::string& dir_path);

bool ListCurrentDir(const std::string& dir_path,
                    std::vector<std::string>* file_list);

bool IsExist(const std::string& path);

bool IsDir(const std::string& path);

bool RemoveLocalFile(const std::string& path);

bool MoveLocalFile(const std::string& src_file,
                   const std::string& dst_file);


void SplitString(const std::string& full, const std::string& delim,
                 std::vector<std::string>* result);

} // namespace io
} // namespace tofu
