/* ====================================================
#   Copyright (C) 2024 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : path.cc
#   Last Modified : 2024-07-16 10:52
#   Describe      : 
#
# ====================================================*/

#include "io/path.h"


#include <stdint.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <pwd.h>
#include <grp.h>
#include <string.h>

namespace tofu {
namespace io {

const int32_t file_op_retry_times = 3;

void SplitStringPath(const std::string& full_path,
                     std::string* dir_part,
                     std::string* file_part) {
    std::string::size_type pos = full_path.rfind("/");
    if (pos != std::string::npos) {
        if (dir_part) {
            *dir_part = full_path.substr(0, pos);
        }
        if (file_part) {
            *file_part = full_path.substr(pos + 1);
        }
    } else {
        if (dir_part) {
            *dir_part = full_path;
        }
    }
}

std::string ConcatStringPath(const std::vector<std::string>& sections,
                             const std::string& delim) {
    std::string file_path;
    if (sections.size() == 0) {
        return file_path;
    }

    for (uint32_t i = 0; i < sections.size() - 1; ++i) {
        file_path += (sections[i] + delim);
    }
    return file_path + sections[sections.size() - 1];
}


std::string GetPathPrefix(const std::string& full_path,
                          const std::string& delim) {
    std::string prefix;
    if (full_path.empty()) {
        return prefix;
    }
    size_t idx = full_path.find(delim, 1);
    if (idx == std::string::npos) {
        return prefix;
    }
    if (idx + 1 != full_path.length()) {
        return full_path.substr(0, idx + 1);
    } else {
        return full_path;
    }
}

bool CreateDirWithRetry(const std::string& dir_path) {
    if (dir_path.length() == 0) {
        return false;
    }
    std::vector<std::string> path_sections;
    SplitString(dir_path, "/", &path_sections);
    bool is_success = true;
    std::string path;
    if (dir_path[0] == '/') {
        path.append("/");
    }
    for (uint32_t d = 0; d < path_sections.size() && is_success; ++d) {
        if (path_sections[d] == ".") {
            continue;
        }
        path += path_sections[d] + "/";
        if (path_sections[d] == "..") {
            continue;
        }

        if (IsExist(path)) {
            continue;
        }

        int i = 0;
        for (; i < file_op_retry_times; i++) {
            if (0 == mkdir(path.c_str(), 0755)) {
                break;
            }
        }
        if (i == file_op_retry_times) {
            is_success = false;
        }
    }
    return is_success;
}

// std::string UidToName(uid_t uid) {
//     struct passwd *temp = NULL;
//     if (NULL == (temp=getpwuid(uid))) {
//         return "";
//     } else {
//         return temp->pw_name;
//     }
// }

// std::string GidToName(gid_t gid) {
//    struct group *temp = NULL;
//    if (NULL == (temp=getgrgid(gid))){
//        return "";
//    } else {
//        return temp->gr_name;
//    }
// }


bool ListCurrentDir(const std::string& dir_path,
                    std::vector<std::string>* file_list) {
    DIR *dir = NULL;
    struct dirent *ptr = NULL;
    dir = opendir(dir_path.c_str());
    if (dir == NULL) {
        closedir(dir);
        return false;
    }
    while ((ptr = readdir(dir)) != NULL) {
        /// if (ptr->d_type == DT_REG) {
        ///     file_list->push_back(ptr->d_name);
        /// }
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            file_list->push_back(ptr->d_name);
        }
    }
    closedir(dir);
    return true;
}

bool IsExist(const std::string& path) {
    return access(path.c_str(), R_OK) == 0;
}

bool IsDir(const std::string& path) {
    if (!IsExist(path)) {
        return false;
    }

    struct stat st;
    if((stat(path.c_str(),&st) == 0)
       && ((st.st_mode & S_IFDIR) != 0)) {
        return true;
    }
    return false;
}

bool RemoveLocalFile(const std::string& path) {
    bool done = false;
    for (int32_t i = 0; i < file_op_retry_times && !done; ++i) {
        done = (remove(path.c_str()) == 0);
    }
    return done;
}

bool MoveLocalFile(const std::string& src_file,
                   const std::string& dst_file) {
    bool done = false;
    for (int32_t i = 0; i < file_op_retry_times && !done; ++i) {
        done = (rename(src_file.c_str(), dst_file.c_str()) == 0);
    }
    return done;
}

void SplitString(const std::string& full, const std::string& delim,
                 std::vector<std::string>* result) {
    result->clear();

    size_t pos_start = 0, pos_end, delim_len = delim.length();
    std::string token;

    while ((pos_end = full.find(delim, pos_start)) != std::string::npos) {
        token = full.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        result->push_back(token);
    }
    result->push_back(full.substr(pos_start));
}

} // namespace io
} // namespace tofu
