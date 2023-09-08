#ifndef __MESH_H__
#define __MESH_H__

#include <vector>
#include "origin.h"
// 模型类

class Mesh
{
public:
	Mesh();
	Mesh(const char *filename); // 根据.obj文件路径导入模型
	~Mesh();
	int nverts();			// 返回模型顶点数量
	int nfaces();			// 返回模型面片数量
	vec3f vert(int i);		// 返回第i个顶点
	vec3i face(int idx);	// 返回第idx个面片
	std::vector<vec3f> verts_;	// 顶点集，每个顶点都是三维向量
	std::vector<vec3i> faces_;	// 面片集
	std::vector<vec3f> normals_; // 法向量集
};

#endif //__MODEL_H__