#pragma once

#include <string>

inline std::string get_asset_path()
{
    return std::string{ ASSET_PATH } + "/";
}