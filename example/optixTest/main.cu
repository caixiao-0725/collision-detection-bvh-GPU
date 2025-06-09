#include "optixLauncher.h"
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#include <sstream>
#include <fstream>
#include <vector>

#include "origin.h"
#include "DeviceHostVector.h"
#include "common.h"

using namespace CXE;

class Mesh {
public:
	DeviceHostVector<vec3f> points;
	DeviceHostVector<vec3u> faces;

	void readObj(const char* filename) {
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
                points.GetHost().push_back(v); // 加入顶点集
            }
            else if (!line.compare(0, 2, "f "))
            { // 如果这一行的前两个字符是“f ”的话，代表是面片数据
                vec3u v;
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
                faces.GetHost().push_back(v); // 把该面片加入模型的面片集
            }
        }

        points.SyncHostSize();
        faces.SyncHostSize();

        points.Allocate(points.GetSize());
        faces.Allocate(faces.GetSize());

        points.ReadToDevice();
        faces.ReadToDevice();
	}
};


int main() {
    // edge
    DeviceHostVector<vec3f> edges_points;
    DeviceHostVector<vec2i> edges_indexs;

    edges_points.Allocate(2);
    edges_indexs.Allocate(1);

    edges_points.GetHost()[0] = vec3f(-0.01, 1, 0);
	edges_points.GetHost()[1] = vec3f(-0.01, -1, 0);

	edges_indexs.GetHost()[0] = vec2i(1, 0);

	edges_points.ReadToDevice();
	edges_indexs.ReadToDevice();
    // model;

	Mesh m_obstacle;

	std::string path = get_asset_path() + "plane/20.obj";
    m_obstacle.readObj(path.c_str());

	OptixLauncher temp;
	temp.init();

    uint32_t vertexStride = 3 * sizeof(float);
    uint32_t posOffset = 0;
    uint32_t vertexCount = m_obstacle.points.GetSize();
    uint32_t indexCount = m_obstacle.faces.GetSize()*3;
    float transform[3][4];
    temp.buildObstacle((void*)m_obstacle.points.GetDevice(), vertexStride, posOffset, vertexCount,
        (void*)m_obstacle.faces.GetDevice(), indexCount, transform);

    temp.launchForEdge((void*)edges_points.GetDevice(), (void*)edges_indexs.GetDevice(), edges_indexs.GetSize());

    HitResult result_cpu[20];
	cudaMemcpy(result_cpu, temp.m_gpuHitResults, sizeof(HitResult) * 20, cudaMemcpyDeviceToHost);

	return 0;
}