#ifndef __MESH_H__
#define __MESH_H__

#include <vector>
#include "origin.h"
// ģ����

class Mesh
{
public:
	Mesh();
	Mesh(const char *filename); // ����.obj�ļ�·������ģ��
	~Mesh();
	int nverts();			// ����ģ�Ͷ�������
	int nfaces();			// ����ģ����Ƭ����
	vec3f vert(int i);		// ���ص�i������
	vec3i face(int idx);	// ���ص�idx����Ƭ
	std::vector<vec3f> verts_;	// ���㼯��ÿ�����㶼����ά����
	std::vector<vec3i> faces_;	// ��Ƭ��
	std::vector<vec3f> normals_; // ��������
};

#endif //__MODEL_H__