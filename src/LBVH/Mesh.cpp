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

// ���캯�������������.obj�ļ�·��
Mesh::Mesh(const char *filename) : verts_(), faces_()
{
    std::ifstream in;
    in.open(filename, std::ifstream::in); // ��.obj�ļ�
    if (in.fail())
        return;
    std::string line;
    while (!in.eof())
    {                           // û�е��ļ�ĩβ�Ļ�
        std::getline(in, line); // ����һ��
        std::istringstream iss(line.c_str());
        char trash;
        if (!line.compare(0, 2, "v "))
        { // �����һ�е�ǰ�����ַ��ǡ�v ���Ļ��������Ƕ�������
            iss >> trash;
            vec3f v; // ���붥������
            for (int i = 0; i < 3; i++)
                iss >> v.raw[i];
            verts_.push_back(v); // ���붥�㼯
        }
        else if (!line.compare(0, 2, "f "))
        { // �����һ�е�ǰ�����ַ��ǡ�f ���Ļ�����������Ƭ����
            vec3i v;
            int iuv, idx; // idx�Ƕ���������itrash������������ʱ�ò�������������ͷ�������
            iss >> trash;
            int i = 0;
            while (iss >> idx)
            {          // ��ȡx/x/x��ʽ
                idx--; // all indices start at 1, not 0
                v.raw[i] = idx;
                ; // �������Ƭ�Ķ��㼯
                i++;
            }
            faces_.push_back(v); // �Ѹ���Ƭ����ģ�͵���Ƭ��
        }
    }

    std::cerr << "v# " << verts_.size() << "   f# " << faces_.size() << std::endl; // �����������Ƭ����
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
