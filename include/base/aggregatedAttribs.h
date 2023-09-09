#ifndef __AGGREGATED_ATTRIBS_H_
#define __AGGREGATED_ATTRIBS_H_

#include <cuda_runtime.h>

	/**
	 * \brief	Constructed on host, copied to device, used in cuda kernels
	 */
	template<int numAttribs>
	class AttribPort {	///< indices to certain attributes
	public:
		__host__ __device__ AttribPort() {}
		__host__ __device__ ~AttribPort() {}
		// selective link copy
		__inline__ __host__ void link(void **_hdAttrs, int *stencil) {
			for (int i = 0; i < numAttribs; i++)
				cudaMemcpy(_ddAttrs + i, _hdAttrs + stencil[i], sizeof(void*), cudaMemcpyHostToHost);
		}
		// continuous link copy
		__inline__ __host__ void link(void **_hdAttrs, int stpos) {
			cudaMemcpy(_ddAttrs, _hdAttrs + stpos, sizeof(void*)*numAttribs, cudaMemcpyHostToHost);
		}
	protected:
		void*	_ddAttrs[numAttribs];
	};

	/**
	 * \brief	Distributer of ports, managed on host
	 * \tparam numAttribs 
	 * \tparam numPorts 
	 */
	template<int numAttribs, int numPorts = 1>
	class AttribConnector {
	public:
		AttribConnector() {}
		~AttribConnector() {}

	protected:
		/// manage data
		void*	_attribs[numAttribs];		///< cuda allocated arrays
		/// distribute ports
		void*	_ports[numPorts];
	};

#endif