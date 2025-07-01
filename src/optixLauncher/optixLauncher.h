#ifndef _OPTIX_LAUNCHER_H_
#define _OPTIX_LAUNCHER_H_
#include "optix.h"
#include "cuda.h"
#include "stdint.h"
#include <cuda_runtime.h>
#include "edgeTriangles.h"
#include "origin.h"
#include "bv.h"
#include "DeviceHostVector.h"

typedef unsigned int uint;
#define OPTIX_CHECK(call)                                                 \
	do                                                                    \
	{                                                                     \
		OptixResult res = call;                                           \
		if (res != OPTIX_SUCCESS)                                         \
		{                                                                 \
			std::cerr << optixGetErrorName(res) << ": "                   \
					  << "Optix call '" << #call                          \
					  << "' failed: (" __FILE__ ":" << __LINE__ << ")\n"; \
		}                                                                 \
	} while (0)

#define CUDA_CHECK(call)                                                     \
	do                                                                       \
	{                                                                        \
		cudaError_t error = call;                                            \
		if (error != cudaSuccess)                                            \
		{                                                                    \
			std::cerr << "CUDA call (" << #call << " ) failed with error: '" \
					  << cudaGetErrorString(error)                           \
					  << "' (" __FILE__ << ":" << __LINE__ << ")\n";         \
		}                                                                    \
	} while (0)

#define OPTIX_CHECK_LOG(call)                                                     \
	do                                                                            \
	{                                                                             \
		OptixResult res = call;                                                   \
		const size_t sizeof_log_returned = sizeof_log;                            \
		sizeof_log = sizeof(log); /* reset sizeof_log for future calls */         \
		if (res != OPTIX_SUCCESS)                                                 \
		{                                                                         \
			std::cerr << optixGetErrorName(res) << ": "                           \
					  << "Optix call '" << #call                                  \
					  << "' failed: (" __FILE__ ":" << __LINE__ << ")\nLog:\n"    \
					  << log                                                      \
					  << (sizeof_log_returned > sizeof(log) ? "<TRUNCATED>" : "") \
					  << "\n";                                                    \
		}                                                                         \
	} while (0)

struct OptixTriangleLauncherInfo {
	uint32_t buildInputFlags = OPTIX_GEOMETRY_FLAG_NONE;
	OptixBuildInput triangleInput = {};
	//OptixBuildInput iasBuildInput = {};
	CUdeviceptr vertexBuffer = 0;
	CUdeviceptr indexBuffer = 0;
	CUdeviceptr tempBuffer = 0;
	CUdeviceptr gasOutputBuffer = 0;
	//CUdeviceptr iasOutputBuffer = 0;
	//CUdeviceptr iasInstances = 0;
	size_t gasOutputBufferSize = 0;
	//size_t iasOutputBufferSize = 0;
	size_t tempBufferSize = 0;
};

struct OptixAABBLauncherInfo {
	uint32_t buildInputFlags = OPTIX_GEOMETRY_FLAG_NONE;
	OptixBuildInput AABBInput = {};
	//OptixBuildInput iasBuildInput = {};
	CUdeviceptr AABBBuffer = 0;
	CUdeviceptr numAABB = 0;

	CUdeviceptr tempBuffer = 0;
	CUdeviceptr gasOutputBuffer = 0;

	size_t gasOutputBufferSize = 0;
	size_t tempBufferSize = 0;
};



class OptixLauncher
{
public:
	OptixLauncher();
	~OptixLauncher();


	void init();
	void createContext();
	void createTriangleModule();
	void createAABBModule();

	OptixProgramGroup createRaygenProgramGroup();
	OptixProgramGroup createRayPointgenProgramGroup();
	OptixProgramGroup createMissProgramGroup();

	OptixProgramGroup createHitgroupProgramGroup(bool useAnyhit, const char* entryName);

	OptixProgramGroup createAABBHitgroupProgramGroup(bool useAnyhit,const char* anyhitName,const char* intersectName);

	OptixPipeline linkPipeline(OptixProgramGroup hitgroupProgGroup);
	void setupShaderBindingTable(OptixShaderBindingTable& sbt, float radius, OptixProgramGroup hitgroupProgGroup);

	OptixDeviceContext m_context;
	OptixPipelineCompileOptions m_pipeline_compile_options;
	OptixModule m_module;
	OptixModule m_builtinSphereModule;
	OptixProgramGroup m_raygenProgGroup;
	OptixProgramGroup m_missProgGroup;
	OptixProgramGroup m_obstacleHitgroupProgGroup;
	OptixProgramGroup m_clothHitgroupProgGroup;
	OptixPipeline m_obstaclePipeline;
	OptixPipeline m_clothPipeline;
	OptixShaderBindingTable m_obstacleSbt;
	OptixShaderBindingTable m_clothSbt;

	OptixTriangleLauncherInfo m_obstacleTriangleLauncherInfo;
	OptixAABBLauncherInfo m_obstacleAABBLauncherInfo;

	OptixTraversableHandle m_obstacleGASHandle;
	OptixTraversableHandle m_obstacleIASHandle;

	CUstream m_stream;

	Params m_cpuParams;
	CUdeviceptr m_gpuParams;

	void buildTriangleObstacle(const void* verts,
		unsigned int strideInBytes,
		unsigned int posOffsetInBytes,
		unsigned int vertCnt,
		const void* index,
		unsigned int indexCnt,
		float transform[3][4]);

	void buildTriangleGeometry(OptixTriangleLauncherInfo& info,
		OptixTraversableHandle& gasHandle,
		OptixTraversableHandle& iasHandle,
		const void* verts,
		uint32_t strideInBytes,
		uint32_t posOffsetInBytes,
		uint32_t vertCnt,
		const void* index,
		uint32_t indexCnt,
		float transform[3][4]);

	void buildTriangleGeometryWithGPUData(OptixTriangleLauncherInfo& info,
		OptixTraversableHandle& gasHandle,
		OptixTraversableHandle& iasHandle,
		uint32_t vertCnt,
		uint32_t indexCnt,
		float transform[3][4]);

	void buildAABBObstacle(const vec3f* verts,
		const vec3i* index,
		unsigned int faceCnt,
		const float thickness,
		float transform[3][4]);

	void buildAABBObstacleFromPoint(const vec3f* points,
		unsigned int pointCnt,
		const float thickness,
		float transform[3][4]);

	void OptixLauncher::buildAABBGeometryFromPoint(OptixAABBLauncherInfo& info,
		OptixTraversableHandle& gasHandle,
		OptixTraversableHandle& iasHandle,
		const vec3f* gpuPointBuffer,
		uint32_t pointCnt,
		const float thickness,
		float transform[3][4]);

	void buildAABBGeometry(OptixAABBLauncherInfo& info,
		OptixTraversableHandle& gasHandle,
		OptixTraversableHandle& iasHandle,
		const vec3f* gpuVertexBuffer,
		const vec3i* gpuIndexBuffer,
		uint32_t faceCnt,
		const float thickness,
		float transform[3][4]);

	void buildAABBGeometryWithGPUData(OptixAABBLauncherInfo& info,
		OptixTraversableHandle& gasHandle,
		OptixTraversableHandle& iasHandle,
		uint32_t aabbCnt,
		float transform[3][4]);

	void launchForEdge(void* gpuVerts, void* edges, const uint numEdges);

	void launchForVert(void* gpuVerts, const uint numVerts);

	char log[2048];  // For error reporting from OptiX creation functions

	float m_particleSphereRadius = 0.001f;

	Lczx::DeviceHostVector<int> m_cdIndex;
	Lczx::DeviceHostVector<int> m_cdBuffer;

	int m_type = 2; // 0.edge - triangle primitive    1. edge - aabb primitive   2. point - aabb primitive
};



void setVertexBuffer(float3* gpuVertexBuffer, const void* verts, uint strideInBytes, uint posOffsetInBytes, uint numVertices);

void setAABBBuffer(AABB* AABBBuffer, const vec3f* verts, const vec3i* indexs, const float thickness, uint numAABB);

void setAABBBufferFromPoints(AABB* AABBBuffer, const vec3f* verts, const float thickness, uint numAABB);

#endif