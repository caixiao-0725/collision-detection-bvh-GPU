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

using namespace Lczx;

class Mesh {
public:
	DeviceHostVector<vec3f> points;
	DeviceHostVector<vec3i> faces;

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

void loadobj(std::string filename, std::vector<vec3f>& position, std::vector<vec3i>& face, std::vector<vec2i>& edge) {
    std::ifstream in;
    in.open(filename, std::ifstream::in);
    if (in.fail())
        return;
    std::string line;
    while (!in.eof())
    {
        std::getline(in, line);
        std::istringstream iss(line.c_str());
        char trash;
        if (!line.compare(0, 2, "v "))
        {
            iss >> trash;
            vec3f v;
            for (int i = 0; i < 3; i++)
                iss >> v.raw[i];
            position.push_back(v);
        }
        else if (!line.compare(0, 2, "f "))
        {
            int iuv, idx;
            iss >> trash;
            vec3i f;
            int count = 0;
            std::string vertex;
            while (iss >> vertex)
            {
                std::istringstream vss(vertex);
                std::string index;
                std::getline(vss, index, '/');
                std::istringstream(index) >> idx;
                f.raw[count] = idx - 1;
                count++;
            }
            face.push_back(f);
        }
    }

    std::set<std::pair<int, int>> edges;
    for (int f = 0; f < face.size();f++) {
        for (int i = 0;i < 3;i++) {
            int v1 = face[f].raw[i];
            int v2 = face[f].raw[(i + 1) % 3];
            if (v1 > v2) std::swap(v1, v2);
            edges.insert(std::make_pair(v1, v2));
        }
    }

    for (const auto& e : edges) {
        edge.push_back(vec2i(e.first, e.second));
    }

    std::cerr << "[Mesh] vert: " << position.size() << "   face: " << face.size() << std::endl;
}


int main() {


    //// edge
    //DeviceHostVector<vec3f> edges_points;
    //DeviceHostVector<vec2i> edges_indexs;
    //
    //edges_points.Allocate(2);
    //edges_indexs.Allocate(1);
    //
    //edges_points.GetHost()[0] = vec3f(0, 1, 0);
	//edges_points.GetHost()[1] = vec3f(0, -1, 0);
    //
	//edges_indexs.GetHost()[0] = vec2i(1, 0);
    //
	//edges_points.ReadToDevice();
	//edges_indexs.ReadToDevice();
    //// model;
    //
	//Mesh m_obstacle;
    //
	//std::string path = get_asset_path() + "plane/20.obj";
    //m_obstacle.readObj(path.c_str());
    //
	//OptixLauncher optix;
    //optix.m_type = 1;
    //optix.init();
    //
    //float transform[3][4];
    //optix.buildAABBObstacle(m_obstacle.points.GetDevice(), m_obstacle.faces.GetDevice(), m_obstacle.faces.GetSize(), 0.001f, transform);
    //optix.launchForEdge((void*)edges_points.GetDevice(), (void*)edges_indexs.GetDevice(), edges_indexs.GetSize());


    std::string mesh_file = get_asset_path() + "models/0.obj";
    std::vector<vec3f> position;
    std::vector<vec3i> face;
    std::vector<vec2i> edge;
    loadobj(mesh_file, position, face, edge);
    DeviceHostVector<vec3f> d_position;
    d_position.Allocate(position.size());
    d_position.SetDeviceHost(position.size(), position.data());
    DeviceHostVector<vec3i> d_face;
    d_face.Allocate(face.size());
    d_face.SetDeviceHost(face.size(), face.data());
    DeviceHostVector<vec2i> d_edge;
    d_edge.Allocate(edge.size());
    d_edge.SetDeviceHost(edge.size(), edge.data());


    OptixLauncher optix;
    optix.m_type = 1;
    optix.init();

    float transform[3][4];
    optix.buildAABBObstacle(d_position.GetDevice(), d_face.GetDevice(), d_face.GetSize(), 0.001f, transform);
    optix.launchForEdge((void*)d_position.GetDevice(), (void*)d_edge.GetDevice(), d_edge.GetSize());
    
    //optix.buildTriangleObstacle((void*)m_obstacle.points.GetDevice(), vertexStride, posOffset, vertexCount,
    //(void*)m_obstacle.faces.GetDevice(), indexCount, transform);

    optix.m_cdIndex.ReadToHost();
    optix.m_cdBuffer.ReadToHost();
    int sum = 0;
    for (size_t i = 0; i < optix.m_cdIndex.GetSize(); i++) {
        sum += optix.m_cdIndex.GetHost()[i];
    }
    std::cout << "Collision Count: " << sum << std::endl;

	return 0;
}

int main0() {
    const int N = 100000;
    const float R = 0.001f;

    printf("Generating Data...\n");
    DeviceHostVector<vec3f> points;
    points.Allocate(N);
    srand(1);
    for (size_t i = 0; i < N; i++) {
        points.GetHost()[i] = vec3f(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
        //points.GetHost()[i] = vec3f(i*R*1.1, i*R*1.1, i*R*1.1);
    }

    points.ReadToDevice();

    OptixLauncher optix;
    optix.init();

    uint32_t pointCount = points.GetSize();
    float transform[3][4];

    optix.buildAABBObstacleFromPoint(points, pointCount, R*2, transform);

    optix.launchForVert((void*)points.GetDevice(), points.GetSize());

    optix.m_cdIndex.ReadToHost();
	optix.m_cdBuffer.ReadToHost();
    int sum = 0;
	for (size_t i = 0; i < optix.m_cdIndex.GetSize(); i++) {
		sum += optix.m_cdIndex.GetHost()[i];
	}
	std::cout << "Collision Count: " << sum << std::endl;
    return 0;
}
