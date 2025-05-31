#ifndef DEVICEHOSTVECTOR_H
#define DEVICEHOSTVECTOR_H

#include "Common.h"

namespace CXE
{
	template <typename T>
	class DeviceHostVector
	{
	public:
		// Device memory pointer
		T *ptr = nullptr;

		// Host memory
		std::vector<T> hostVector;

		// Allocated memory size
		size_t size = 0;

	public:
		// Copy-swap implementation
		// Swap function of this class
		// Should be updated if you add new member variables
		friend void swap(DeviceHostVector<T> &m1, DeviceHostVector<T> &m2) noexcept
		{
			std::swap(m1.ptr, m2.ptr); // Swap resource pointer to avoid multi-release
			std::swap(m1.size, m2.size);
			std::swap(m1.hostVector, m2.hostVector);
		}

		// Default constructor
		DeviceHostVector() = default;

		// Delete default copy constructor
		DeviceHostVector(const DeviceHostVector &obj) = delete;

		// Custom move constructor
		DeviceHostVector(DeviceHostVector &&other) noexcept { swap(*this, other); }

		// Free memory and reset members
		~DeviceHostVector() { Release(); }

		// Delete default copy assignment operator
		DeviceHostVector &operator=(const DeviceHostVector &obj) = delete;

		// Custom move assignment operator
		DeviceHostVector &operator=(DeviceHostVector &&other) noexcept
		{
			swap(*this, other);
			return *this;
		}

	public: // Memory management interfaces
		// Convert to pointer implicitly
		operator T *() const { return static_cast<T *>(ptr); }

		// Get host memory vector
		const std::vector<T> &GetHost() const { return hostVector; }

		// Get host memory vector
		std::vector<T> &GetHost() { return hostVector; }

		// Get pointer explicitly
		T *GetDevice() const { return static_cast<T *>(ptr); }

		// Get size
		size_t GetSize() const { return size; }

		// Allocate and set capacity
		void Allocate(size_t newSize);

		// Allocate and set capacity and intialize conteng
		void Allocate(size_t newSize, const T *hostData);

		// Clear member without release memory pointer
		void Reset();

		// Free memory and reset members
		void Release();

		// CPU mem set
		void SetHost(size_t newSize, T value);

		// CPU mem set
		void SetHost(size_t newSize, const T *hostData);

		// CPU & GPU mem set
		void SetDeviceHost(size_t newSize, T value);

		// Copy to device & host memory
		void SetDeviceHost(size_t newSize, const T *hostData);

		// Copy host to device memory
		void ReadToDevice() const;

		// Copy device to host memory
		void ReadToHost() const;

		void SyncHostSize() { size = hostVector.size(); }
	};
}

#endif
