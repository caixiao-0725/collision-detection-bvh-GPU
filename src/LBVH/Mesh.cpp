#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "Mesh.h"
#include <omp.h>

#include <ctime>


Mesh::Mesh()
{
}

// 构造函数，输入参数是.obj文件路径
Mesh::Mesh(const char *filename) : verts_(), faces_()
{
    std::ifstream in;
    in.open(filename, std::ifstream::in); // 打开.obj文件
    if (in.fail())
        return;
    std::string line;
    while (!in.eof())
    {                           // 没有到文件末尾的话
        std::getline(in, line); // 读入一行
        std::istringstream iss(line.c_str());
        char trash;
        if (!line.compare(0, 2, "v "))
        { // 如果这一行的前两个字符是“v ”的话，代表是顶点数据
            iss >> trash;
            vec3f v; // 读入顶点坐标
            for (int i = 0; i < 3; i++)
                iss >> v.raw[i];
            verts_.push_back(v); // 加入顶点集
        }
        else if (!line.compare(0, 2, "f "))
        { // 如果这一行的前两个字符是“f ”的话，代表是面片数据
            vec3i v;
            int iuv, idx; // idx是顶点索引，itrash用来读我们暂时用不到的纹理坐标和法线向量
            iss >> trash;
            int i = 0;
            while (iss >> idx)
            {          // 读取x/x/x格式
                idx--; // all indices start at 1, not 0
                v.raw[i] = idx;
                ; // 加入该面片的顶点集
                i++;
            }
            faces_.push_back(v); // 把该面片加入模型的面片集
        }
    }

    std::cerr << "v# " << verts_.size() << "   f# " << faces_.size() << std::endl; // 输出顶点与面片数量
}

Mesh::~Mesh()
{
}

int Mesh::nverts()
{
    return (int)verts_.size();
}

int Mesh::nfaces()
{
    return (int)faces_.size();
}

vec3i Mesh::face(int i)
{
    return faces_[i];
}

vec3f Mesh::vert(int i)
{
    return verts_[i];
}
