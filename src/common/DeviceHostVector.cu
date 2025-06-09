#include "DeviceHostVector.h"
#include "HelperCuda.h"
#include "origin.h"
#include "bv.h"
//#include "collisionUtils.cuh"
namespace CXE
{
    // Explicit Instantiation
	template class DeviceHostVector<char>;
	template class DeviceHostVector<char2>;
	template class DeviceHostVector<char3>;
	template class DeviceHostVector<char4>;

	template class DeviceHostVector<unsigned char>;
	template class DeviceHostVector<uchar2>;
	template class DeviceHostVector<uchar3>;
	template class DeviceHostVector<uchar4>;

	template class DeviceHostVector<short>;
	template class DeviceHostVector<short2>;
	template class DeviceHostVector<short3>;
	template class DeviceHostVector<short4>;

	template class DeviceHostVector<unsigned short>;
	template class DeviceHostVector<ushort2>;
	template class DeviceHostVector<ushort3>;
	template class DeviceHostVector<ushort4>;

	template class DeviceHostVector<int>;
	template class DeviceHostVector<int2>;
	template class DeviceHostVector<int3>;
	template class DeviceHostVector<int4>;

	template class DeviceHostVector<unsigned int>;
	template class DeviceHostVector<uint2>;
	template class DeviceHostVector<uint3>;
	template class DeviceHostVector<uint4>;

	template class DeviceHostVector<long>;
	template class DeviceHostVector<long2>;
	template class DeviceHostVector<long3>;
	template class DeviceHostVector<long4>;

	template class DeviceHostVector<unsigned long>;
	template class DeviceHostVector<ulong2>;
	template class DeviceHostVector<ulong3>;
	template class DeviceHostVector<ulong4>;

	template class DeviceHostVector<long long>;
	template class DeviceHostVector<longlong2>;
	template class DeviceHostVector<longlong3>;
	template class DeviceHostVector<longlong4>;

	template class DeviceHostVector<unsigned long long>;
	template class DeviceHostVector<ulonglong2>;
	template class DeviceHostVector<ulonglong3>;
	template class DeviceHostVector<ulonglong4>;

	template class DeviceHostVector<float>;
	template class DeviceHostVector<float2>;
	template class DeviceHostVector<float3>;
	template class DeviceHostVector<float4>;

	template class DeviceHostVector<double>;
	template class DeviceHostVector<double2>;
	template class DeviceHostVector<double3>;
	template class DeviceHostVector<double4>;

    template class DeviceHostVector<vec4f>;
    template class DeviceHostVector<vec3f>;
    template class DeviceHostVector<vec3u>;
    template class DeviceHostVector<vec2f>;
    template class DeviceHostVector<vec3i>;
    template class DeviceHostVector<vec2i>;
    template class DeviceHostVector<AABB>;
    template class DeviceHostVector<bvhNode>;
    template class DeviceHostVector<bvhNodeV1>;
    template class DeviceHostVector<bvhNodeV2>;
    //template class DeviceHostVector<Utils::CollsionPair>;

    // Allocate and set capacity
	template <typename T>
    void DeviceHostVector<T>::Allocate(size_t newSize)
    {
        if (ptr != nullptr)
        {
            checkCudaErrors(cudaFree(ptr));
        }
        checkCudaErrors(cudaMalloc(&ptr, newSize * sizeof(T)));
        size = newSize;
        hostVector.resize(newSize);
    }

    // Allocate and set capacity
	template <typename T>
    void DeviceHostVector<T>::Allocate(size_t newSize, const T* hostData)
    {
        Allocate(newSize);
        SetDeviceHost(newSize, hostData);
    }

    // Clear member without release memory pointer
	template <typename T>
    void DeviceHostVector<T>::Reset()
    {
        ptr = nullptr;
        size = 0;
        hostVector.resize(0);
    }

    // Free memory and reset members
	template <typename T>
    void DeviceHostVector<T>::Release()
    {
        if (ptr != nullptr)
        {
            checkCudaErrors(cudaFree(ptr));
        }
        Reset();
    }

    // CPU mem set
    template <typename T>
    void DeviceHostVector<T>::SetHost(size_t newSize, T value)
    {
        std::fill(hostVector.begin(), hostVector.begin() + newSize, value);
    }

    // CPU mem set
    template <typename T>
    void DeviceHostVector<T>::SetHost(size_t newSize, const T* hostData)
    {
        std::copy(hostData, hostData + newSize, hostVector.begin());
    }

    // GPU mem set
	template <typename T>
    void DeviceHostVector<T>::SetDeviceHost(size_t newSize, T value)
    {
        SetHost(newSize, value);
        ReadToDevice();
    }

    // Copy to device memory
	template <typename T>
    void DeviceHostVector<T>::SetDeviceHost(size_t newSize, const T* hostData)
    {
        checkCudaErrors(cudaMemcpy(ptr, hostData, newSize * sizeof(T), cudaMemcpyHostToDevice));
        std::copy(hostData, hostData + newSize, hostVector.begin());
    }

    // Read to device memory
	template <typename T>
    void DeviceHostVector<T>::ReadToDevice() const
    {
        checkCudaErrors(cudaMemcpy(ptr, hostVector.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Read to host memory
	template <typename T>
    void DeviceHostVector<T>::ReadToHost() const
    {
        checkCudaErrors(cudaMemcpy((void*)hostVector.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
    }
}
