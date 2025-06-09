#ifndef _OPTIX_LAUNCHER_H_
#define _OPTIX_LAUNCHER_H_
#include "optix.h"
#include "cuda.h"
#include "stdint.h"
#include <cuda_runtime.h>
#include "edgeTriangles.h"

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

struct OptixLauncherInfo {
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

class OptixLauncher
{
public:
	OptixLauncher();
	~OptixLauncher();


	void init();
	void createContext();
	void createModule();

	OptixProgramGroup createRaygenProgramGroup();
	OptixProgramGroup createMissProgramGroup();
	OptixProgramGroup createHitgroupProgramGroup(bool useAnyhit,const char* entryName);

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

	OptixLauncherInfo m_obstacleLauncherInfo;

	OptixTraversableHandle m_obstacleGASHandle;
	OptixTraversableHandle m_obstacleIASHandle;

	CUstream m_stream;

	Params m_cpuParams;
	CUdeviceptr m_gpuParams;

	void buildObstacle(const void* verts,
		unsigned int strideInBytes,
		unsigned int posOffsetInBytes,
		unsigned int vertCnt,
		const void* index,
		unsigned int indexCnt,
		float transform[3][4]);

	void buildGeometry(OptixLauncherInfo& info,
		OptixTraversableHandle& gasHandle,
		OptixTraversableHandle& iasHandle,
		const void* verts,
		uint32_t strideInBytes,
		uint32_t posOffsetInBytes,
		uint32_t vertCnt,
		const void* index,
		uint32_t indexCnt,
		float transform[3][4]);

	void buildGeometryWithGPUData(OptixLauncherInfo& info,
		OptixTraversableHandle& gasHandle,
		OptixTraversableHandle& iasHandle,
		uint32_t vertCnt,
		uint32_t indexCnt,
		float transform[3][4]);

	void launchForEdge(void* gpuVerts, void* edges, const uint numEdges);

	char log[2048];  // For error reporting from OptiX creation functions

	float m_particleSphereRadius = 0.001f;

	HitResult* m_gpuHitResults;
};



void setVertexBuffer(float3* gpuVertexBuffer, const void* verts, uint strideInBytes, uint posOffsetInBytes, uint numVertices);


#endif