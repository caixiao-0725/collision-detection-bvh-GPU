#include "optixLauncher.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>

#include "optix_function_table_definition.h"
#include "optix_stack_size.h"
#include "optix_stubs.h"
#include <array>
#include <map>
#include <vector>
#include <string>
#include "edgeTriangles.h"
#include "common.h"


template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

template <typename IntegerType>
IntegerType roundUp(IntegerType x, IntegerType y)
{
	return ((x + y - 1) / y) * y;
}


static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /* cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

struct PtxSourceCache
{
	std::map<std::string, std::string*> map;
	~PtxSourceCache()
	{
		for (std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end(); ++it)
			delete it->second;
	}
};

static PtxSourceCache g_ptxSourceCache;

static bool readSourceFile(std::string& str, const std::string& filename)
{
	// Try to open file
	std::ifstream file(filename.c_str(), std::ios::binary);
	if (file.good())
	{
		// Found usable source file
		std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
		str.assign(buffer.begin(), buffer.end());
		return true;
	}
	return false;
}

static void getInputDataFromFile(std::string& ptx, const char* filename)
{
	const std::string sourceFilePath = filename;
	// Try to open source PTX file
	if (!readSourceFile(ptx, filename))
	{
		std::string err = "Couldn't open source file " + sourceFilePath;
		throw std::runtime_error(err.c_str());
	}
}


// log: (Optional) pointer to compiler log string. If *log == NULL there is no output. Only valid until the next getInputData call
const char* getInputData(const char* filename, size_t& dataSize, const char** log = NULL)
{
	if (log)
		*log = NULL;

	std::string* ptx, cu;
	std::string key = std::string(filename);
	std::map<std::string, std::string*>::iterator elem = g_ptxSourceCache.map.find(key);

	if (elem == g_ptxSourceCache.map.end())
	{
		ptx = new std::string();
		getInputDataFromFile(*ptx, filename);
		g_ptxSourceCache.map[key] = ptx;
	}
	else
	{
		ptx = elem->second;
	}
	dataSize = ptx->size();
	return ptx->c_str();
}

OptixLauncher::OptixLauncher() {
	m_context = nullptr;

	m_module = nullptr;
	m_builtinSphereModule = nullptr;

	m_raygenProgGroup = nullptr;
	m_missProgGroup = nullptr;
	m_obstacleHitgroupProgGroup = nullptr;
	m_clothHitgroupProgGroup = nullptr;

	m_obstaclePipeline = nullptr;
	m_clothPipeline = nullptr;
}

OptixLauncher::~OptixLauncher() {
	OPTIX_CHECK(optixDeviceContextDestroy(m_context));

	CUDA_CHECK(cudaStreamDestroy(m_stream));
}

// Initialize CUDA and create OptiX context
void OptixLauncher::createContext()
{
	CUcontext cuCtx = 0;  // zero means take the current context
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_log_cb;
	options.logCallbackLevel = 4;
#ifdef _DEBUG
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_context));
}

void OptixLauncher::createTriangleModule() {
	m_pipeline_compile_options = {};

	OptixModuleCompileOptions module_compile_options = {};
	module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

	m_pipeline_compile_options.usesMotionBlur = false;
	m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;//OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	m_pipeline_compile_options.numPayloadValues = 5; //NUM_PAYLOAD_VALUES;
	m_pipeline_compile_options.numAttributeValues = 0; //NUM_ATTRIBUTE_VALUES;
	m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
	m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
	m_pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

	size_t sizeof_log = sizeof(log);

	size_t optixSpherePtxSize = 0;
	std::string fullPath = get_asset_path() + "ptx/cuda_compile_ptx_1_generated_edgeTriangles.cu.ptx";
	std::cout << "PTX: " << fullPath << std::endl;
	const char* optixSpherePtx = getInputData(fullPath.c_str(), optixSpherePtxSize);

	OPTIX_CHECK_LOG(optixModuleCreate(m_context,
		&module_compile_options,
		&m_pipeline_compile_options,
		optixSpherePtx,
		optixSpherePtxSize,
		log,
		&sizeof_log,
		&m_module));

	OptixBuiltinISOptions builtinISOptions = {};
	builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
	builtinISOptions.usesMotionBlur = 0;
	builtinISOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
	OPTIX_CHECK(optixBuiltinISModuleGet(m_context,
		&module_compile_options,
		&m_pipeline_compile_options,
		&builtinISOptions,
		&m_builtinSphereModule));
}

void OptixLauncher::createAABBModule() {
	m_pipeline_compile_options = {};

	OptixModuleCompileOptions module_compile_options = {};
	module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

	m_pipeline_compile_options.usesMotionBlur = false;
	m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;//OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	m_pipeline_compile_options.numPayloadValues = 5; //NUM_PAYLOAD_VALUES;
	m_pipeline_compile_options.numAttributeValues = 3; //NUM_ATTRIBUTE_VALUES;
	m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
	m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
	m_pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

	size_t sizeof_log = sizeof(log);

	size_t optixPtxSize = 0;
	std::string fullPath = get_asset_path() + "ptx/cuda_compile_ptx_1_generated_edgeTriangles.cu.ptx";
	std::cout << "PTX: " << fullPath << std::endl;
	const char* optixPtx = getInputData(fullPath.c_str(), optixPtxSize);

	OPTIX_CHECK_LOG(optixModuleCreate(m_context,
		&module_compile_options,
		&m_pipeline_compile_options,
		optixPtx,
		optixPtxSize,
		log,
		&sizeof_log,
		&m_module));

	//OptixBuiltinISOptions builtinISOptions = {};
	//builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_CUSTOM;
	//builtinISOptions.usesMotionBlur = 0;
	//builtinISOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
	//OPTIX_CHECK(optixBuiltinISModuleGet(m_context,
	//	&module_compile_options,
	//	&m_pipeline_compile_options,
	//	&builtinISOptions,
	//	&m_builtinSphereModule));
}

void OptixLauncher::init()
{
	// Initialize CUDA
	CUDA_CHECK(cudaFree(0));
	CUDA_CHECK(cudaStreamCreate(&m_stream));
	
	createContext();
	if (m_type == 0) // AABB primitive
		createAABBModule();
	else // triangle primitive
		createTriangleModule();

	m_raygenProgGroup = createRaygenProgramGroup();
	m_missProgGroup = createMissProgramGroup();
	if (m_type == 0) // AABB primitive
		m_obstacleHitgroupProgGroup = createAABBHitgroupProgramGroup(true, "__anyhit__ch", "__intersection__is");
	else // triangle primitive
		m_obstacleHitgroupProgGroup = createHitgroupProgramGroup(true,"__anyhit__ch");
	//m_obstacleHitgroupProgGroup = createHitgroupProgramGroup(false, "__closesthit__ch");
	m_obstaclePipeline = linkPipeline(m_obstacleHitgroupProgGroup);

	setupShaderBindingTable(m_obstacleSbt, m_particleSphereRadius, m_obstacleHitgroupProgGroup);

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_gpuParams), sizeof(Params)));
	CUDA_CHECK(cudaMalloc((void**)&m_gpuHitResults, sizeof(HitResult)*100));
}


OptixProgramGroup OptixLauncher::createHitgroupProgramGroup(bool useAnyhit, const char* entryName)
{
	OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

	size_t sizeof_log = sizeof(log);

	OptixProgramGroup hitgroup_prog_group;

	if (useAnyhit)
	{
		OptixProgramGroupDesc hitgroup_prog_group_desc = {};
		hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleAH = m_module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = entryName;
		hitgroup_prog_group_desc.hitgroup.moduleCH = nullptr;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
		hitgroup_prog_group_desc.hitgroup.moduleIS = nullptr;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context,
			&hitgroup_prog_group_desc,
			1,  // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&hitgroup_prog_group));
	}
	else
	{
		OptixProgramGroupDesc hitgroup_prog_group_desc = {};
		hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleAH = nullptr;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
		hitgroup_prog_group_desc.hitgroup.moduleCH = m_module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = entryName;
		hitgroup_prog_group_desc.hitgroup.moduleIS = nullptr;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context,
			&hitgroup_prog_group_desc,
			1,  // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&hitgroup_prog_group));
	}

	return hitgroup_prog_group;
}

OptixProgramGroup OptixLauncher::createAABBHitgroupProgramGroup(bool useAnyhit, const char* anyhitName, const char* intersectName)
{
	OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

	size_t sizeof_log = sizeof(log);

	OptixProgramGroup hitgroup_prog_group;

	if (useAnyhit)
	{
		OptixProgramGroupDesc hitgroup_prog_group_desc = {};
		hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleAH = m_module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = anyhitName;
		hitgroup_prog_group_desc.hitgroup.moduleCH = nullptr;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
		hitgroup_prog_group_desc.hitgroup.moduleIS = m_module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = intersectName;
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context,
			&hitgroup_prog_group_desc,
			1,  // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&hitgroup_prog_group));
	}
	else
	{
		OptixProgramGroupDesc hitgroup_prog_group_desc = {};
		hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleAH = nullptr;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
		hitgroup_prog_group_desc.hitgroup.moduleCH = m_module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = anyhitName;
		hitgroup_prog_group_desc.hitgroup.moduleIS = m_module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = intersectName;
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context,
			&hitgroup_prog_group_desc,
			1,  // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&hitgroup_prog_group));
	}

	return hitgroup_prog_group;
}

OptixProgramGroup OptixLauncher::createRaygenProgramGroup()
{
	OptixProgramGroupOptions program_group_options = {};  // Initialize to zeros

	OptixProgramGroupDesc raygen_prog_group_desc = {};  //
	raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	raygen_prog_group_desc.raygen.module = m_module;
	raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg_edge";

	size_t sizeof_log = sizeof(log);

	OptixProgramGroup raygen_prog_group;

	OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context,
		&raygen_prog_group_desc,
		1,  // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		&raygen_prog_group));

	return raygen_prog_group;
}

OptixProgramGroup OptixLauncher::createMissProgramGroup()
{
	OptixProgramGroupOptions program_group_options = {};

	OptixProgramGroupDesc miss_prog_group_desc = {};
	miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	miss_prog_group_desc.miss.module = m_module;
	miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";

	size_t sizeof_log = sizeof(log);

	OptixProgramGroup miss_prog_group;

	OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context,
		&miss_prog_group_desc,
		1,  // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		&miss_prog_group));

	return miss_prog_group;
}

void OptixLauncher::buildTriangleObstacle(const void* verts,
	uint32_t strideInBytes,
	uint32_t posOffsetInBytes,
	uint32_t vertCnt,
	const void* index,
	uint32_t indexCnt,
	float transform[3][4])
{
	buildTriangleGeometry(m_obstacleTriangleLauncherInfo,
		m_obstacleGASHandle,
		m_obstacleIASHandle,
		verts,
		strideInBytes,
		posOffsetInBytes,
		vertCnt,
		index,
		indexCnt,
		transform);
}


void OptixLauncher::buildTriangleGeometry(OptixTriangleLauncherInfo& info,
	OptixTraversableHandle& gasHandle,
	OptixTraversableHandle& iasHandle,
	const void* gpuVertexBuffer,
	uint32_t strideInBytes,
	uint32_t posOffsetInBytes,
	uint32_t vertCnt,
	const void* gpuIndexBuffer,
	uint32_t indexCnt,
	float transform[3][4])
{
	const size_t verticesSizeInBytes = vertCnt * sizeof(float3);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&info.vertexBuffer), verticesSizeInBytes));

	setVertexBuffer((float3*)info.vertexBuffer,
		gpuVertexBuffer,
		strideInBytes,
		posOffsetInBytes,
		vertCnt);

	const size_t indicesSizeInBytes = indexCnt * sizeof(uint);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&info.indexBuffer), indicesSizeInBytes));
	CUDA_CHECK(cudaMemcpy((void*)info.indexBuffer, gpuIndexBuffer, indicesSizeInBytes, cudaMemcpyDeviceToDevice));

	buildTriangleGeometryWithGPUData(info, gasHandle, iasHandle, vertCnt, indexCnt, transform);
}

void OptixLauncher::buildTriangleGeometryWithGPUData(OptixTriangleLauncherInfo& info,
	OptixTraversableHandle& gasHandle,
	OptixTraversableHandle& iasHandle,
	uint32_t vertCnt,
	uint32_t indexCnt,
	float transform[3][4])
{
	info.buildInputFlags = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
	info.triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	info.triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	info.triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
	info.triangleInput.triangleArray.numVertices = vertCnt;
	info.triangleInput.triangleArray.vertexBuffers = &info.vertexBuffer; // todo
	info.triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	info.triangleInput.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;
	info.triangleInput.triangleArray.numIndexTriplets = indexCnt / 3;
	info.triangleInput.triangleArray.indexBuffer = info.indexBuffer;// todo
	info.triangleInput.triangleArray.flags = &info.buildInputFlags;
	info.triangleInput.triangleArray.numSbtRecords = 1; // hit

	// build GAS
	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
	accel_options.motionOptions.numKeys = 1;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes gasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accel_options, &info.triangleInput, 1, &gasBufferSizes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&info.tempBuffer), gasBufferSizes.tempSizeInBytes));
	info.tempBufferSize = gasBufferSizes.tempSizeInBytes;

	// non-compacted output
	size_t compactedSizeOffset = roundUp<size_t>(gasBufferSizes.outputSizeInBytes, 8ull);
	CUdeviceptr gpuTempOutputBufferAndCompactedSize;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpuTempOutputBufferAndCompactedSize), compactedSizeOffset + 8));

	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitProperty.result = (CUdeviceptr)((char*)gpuTempOutputBufferAndCompactedSize + compactedSizeOffset);

	OPTIX_CHECK(optixAccelBuild(m_context,
		0,  // CUDA stream
		&accel_options,
		&info.triangleInput,
		1,  // num build inputs
		info.tempBuffer,
		gasBufferSizes.tempSizeInBytes,
		gpuTempOutputBufferAndCompactedSize, // output buffer
		gasBufferSizes.outputSizeInBytes,
		&gasHandle,
		&emitProperty,  // emitted property list
		1));            // num emitted properties

	size_t compacted_gas_size;
	CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

	if (compacted_gas_size < gasBufferSizes.outputSizeInBytes)
	{
		//m_gpuOutputBuffer.resizeBySizeInBytes(compacted_gas_size);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&info.gasOutputBuffer), compacted_gas_size));
		// use handle as input and output
		OPTIX_CHECK(optixAccelCompact(m_context,
			0,
			gasHandle,
			info.gasOutputBuffer,
			compacted_gas_size,
			&gasHandle));
		CUDA_CHECK(cudaFree((void*)gpuTempOutputBufferAndCompactedSize)); // free temp output
		info.gasOutputBufferSize = compacted_gas_size;
	}
	else
	{
		info.gasOutputBuffer = gpuTempOutputBufferAndCompactedSize;
		info.gasOutputBufferSize = gasBufferSizes.outputSizeInBytes;
	}
}



void OptixLauncher::buildAABBObstacle(const vec3f* verts,
	const vec3i* index,
	uint32_t indexCnt,
	const float thickness,
	float transform[3][4])
{
	buildAABBGeometry(m_obstacleAABBLauncherInfo,
		m_obstacleGASHandle,
		m_obstacleIASHandle,
		verts,
		index,
		indexCnt,
		thickness,
		transform);
}


void OptixLauncher::buildAABBGeometry(OptixAABBLauncherInfo& info,
	OptixTraversableHandle& gasHandle,
	OptixTraversableHandle& iasHandle,
	const vec3f* gpuVertexBuffer,
	const vec3i* gpuIndexBuffer,
	uint32_t faceCnt,
	const float thickness,
	float transform[3][4])
{
	info.numAABB = faceCnt; // each triangle has one AABB
	const size_t AABBSizeInBytes = info.numAABB * sizeof(AABB);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&info.AABBBuffer), AABBSizeInBytes));

	setAABBBuffer((AABB*)info.AABBBuffer,
		gpuVertexBuffer,
		gpuIndexBuffer,
		thickness,
		faceCnt);

	buildAABBGeometryWithGPUData(info, gasHandle, iasHandle, faceCnt, transform);
}

void OptixLauncher::buildAABBGeometryWithGPUData(OptixAABBLauncherInfo& info,
	OptixTraversableHandle& gasHandle,
	OptixTraversableHandle& iasHandle,
	uint32_t aabbCnt,
	float transform[3][4])
{
	info.buildInputFlags = OPTIX_GEOMETRY_FLAG_NONE;
	info.AABBInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	info.AABBInput.customPrimitiveArray.aabbBuffers = &info.AABBBuffer; // todo
	info.AABBInput.customPrimitiveArray.numPrimitives = aabbCnt;
	info.AABBInput.customPrimitiveArray.flags = &info.buildInputFlags;
	info.AABBInput.customPrimitiveArray.numSbtRecords = 1; // hit

	// build GAS
	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE ;
	accel_options.motionOptions.numKeys = 1;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes gasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accel_options, &info.AABBInput, 1, &gasBufferSizes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&info.tempBuffer), gasBufferSizes.tempSizeInBytes));
	info.tempBufferSize = gasBufferSizes.tempSizeInBytes;

	// non-compacted output
	size_t compactedSizeOffset = roundUp<size_t>(gasBufferSizes.outputSizeInBytes, 8ull);
	CUdeviceptr gpuTempOutputBufferAndCompactedSize;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpuTempOutputBufferAndCompactedSize), compactedSizeOffset + 8));

	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitProperty.result = (CUdeviceptr)((char*)gpuTempOutputBufferAndCompactedSize + compactedSizeOffset);

	OPTIX_CHECK(optixAccelBuild(m_context,
		0,  // CUDA stream
		&accel_options,
		&info.AABBInput,
		1,  // num build inputs
		info.tempBuffer,
		gasBufferSizes.tempSizeInBytes,
		gpuTempOutputBufferAndCompactedSize, // output buffer
		gasBufferSizes.outputSizeInBytes,
		&gasHandle,
		&emitProperty,  // emitted property list
		1));            // num emitted properties

	size_t compacted_gas_size;
	CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

	if (compacted_gas_size < gasBufferSizes.outputSizeInBytes)
	{
		//m_gpuOutputBuffer.resizeBySizeInBytes(compacted_gas_size);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&info.gasOutputBuffer), compacted_gas_size));
		// use handle as input and output
		OPTIX_CHECK(optixAccelCompact(m_context,
			0,
			gasHandle,
			info.gasOutputBuffer,
			compacted_gas_size,
			&gasHandle));
		CUDA_CHECK(cudaFree((void*)gpuTempOutputBufferAndCompactedSize)); // free temp output
		info.gasOutputBufferSize = compacted_gas_size;
	}
	else
	{
		info.gasOutputBuffer = gpuTempOutputBufferAndCompactedSize;
		info.gasOutputBufferSize = gasBufferSizes.outputSizeInBytes;
	}
}

void OptixLauncher::launchForEdge(void* gpuVerts, void* edges, const uint numEdges) {


	if ((m_cpuParams.vertexs != gpuVerts) ||
		(m_cpuParams.edgeIndex != edges) ||
		(m_cpuParams.hitResults != m_gpuHitResults) ||
		(m_cpuParams.handle != m_obstacleGASHandle))
	{
		m_cpuParams.vertexs = (float3*)gpuVerts;
		m_cpuParams.edgeIndex = (int2*)edges;
		m_cpuParams.hitResults = m_gpuHitResults;
		m_cpuParams.handle = m_obstacleGASHandle;
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_gpuParams), &m_cpuParams, sizeof(Params), cudaMemcpyHostToDevice));
	}
	OPTIX_CHECK(optixLaunch(m_obstaclePipeline, 0, m_gpuParams, sizeof(Params), &m_obstacleSbt, numEdges, 1, 1));
}


OptixPipeline OptixLauncher::linkPipeline(OptixProgramGroup hitgroupProgGroup)
{
	const uint32_t max_trace_depth = 1;
	OptixProgramGroup program_groups[] = { m_raygenProgGroup, m_missProgGroup, hitgroupProgGroup };

	OptixPipelineLinkOptions pipeline_link_options = {};
	pipeline_link_options.maxTraceDepth = max_trace_depth;

	size_t sizeof_log = sizeof(log);

	OptixPipeline pipeline;

	OPTIX_CHECK_LOG(optixPipelineCreate(m_context,
		&m_pipeline_compile_options,
		&pipeline_link_options,
		program_groups,
		sizeof(program_groups) / sizeof(program_groups[0]),
		log,
		&sizeof_log,
		&pipeline));

	OptixStackSizes stack_sizes = {};
	for (auto& prog_group : program_groups)
	{
		OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
	}

	uint32_t direct_callable_stack_size_from_traversal;
	uint32_t direct_callable_stack_size_from_state;
	uint32_t continuation_stack_size;
	OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
		max_trace_depth,
		0,  // maxCCDepth
		0,  // maxDCDEpth
		&direct_callable_stack_size_from_traversal,
		&direct_callable_stack_size_from_state,
		&continuation_stack_size));
	OPTIX_CHECK(optixPipelineSetStackSize(pipeline,
		direct_callable_stack_size_from_traversal,
		direct_callable_stack_size_from_state,
		continuation_stack_size,
		1));  // maxTraversableDepth

	return pipeline;
}


void OptixLauncher::setupShaderBindingTable(OptixShaderBindingTable& sbt, float radius, OptixProgramGroup hitgroupProgGroup)
{
	sbt = {};

	CUdeviceptr raygen_record;
	const size_t raygen_record_size = sizeof(RayGenSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
	RayGenSbtRecord rg_sbt;
	//rg_sbt.data.rayCollisionRadius = radius;

	OPTIX_CHECK(optixSbtRecordPackHeader(m_raygenProgGroup, &rg_sbt));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record),
		&rg_sbt,
		raygen_record_size,
		cudaMemcpyHostToDevice));

	CUdeviceptr miss_record;
	size_t miss_record_size = sizeof(MissSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
	MissSbtRecord ms_sbt;
	ms_sbt.data = {};
	OPTIX_CHECK(optixSbtRecordPackHeader(m_missProgGroup, &ms_sbt));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record),
		&ms_sbt,
		miss_record_size,
		cudaMemcpyHostToDevice));

	CUdeviceptr hitgroup_record;
	size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
	HitGroupSbtRecord hg_sbt;
	//hg_sbt.data.rayCollisionRadius = radius;
	OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupProgGroup, &hg_sbt));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record),
		&hg_sbt,
		hitgroup_record_size,
		cudaMemcpyHostToDevice));

	sbt.raygenRecord = raygen_record;
	sbt.missRecordBase = miss_record;
	sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
	sbt.missRecordCount = 1;
	sbt.hitgroupRecordBase = hitgroup_record;
	sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
	sbt.hitgroupRecordCount = 1;
}