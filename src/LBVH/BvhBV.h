#ifndef __BVH_BV_H_
#define __BVH_BV_H_

#include "bv.h"
#include "aggregatedAttribs.h"

class BvhBvCompletePort : public AttribPort<6> {
	public:
		__host__ __device__ BvhBvCompletePort() {}
		__host__ __device__ ~BvhBvCompletePort() {}

		__device__ float& minx(int i) { return ((float*)_ddAttrs[MINX])[i]; }
		__device__ float& miny(int i) { return ((float*)_ddAttrs[MINY])[i]; }
		__device__ float& minz(int i) { return ((float*)_ddAttrs[MINZ])[i]; }
		__device__ float& maxx(int i) { return ((float*)_ddAttrs[MAXX])[i]; }
		__device__ float& maxy(int i) { return ((float*)_ddAttrs[MAXY])[i]; }
		__device__ float& maxz(int i) { return ((float*)_ddAttrs[MAXZ])[i]; }

		__device__ float getminx(int i) const { return ((float*)_ddAttrs[MINX])[i]; }
		__device__ float getminy(int i) const { return ((float*)_ddAttrs[MINY])[i]; }
		__device__ float getminz(int i) const { return ((float*)_ddAttrs[MINZ])[i]; }
		__device__ float getmaxx(int i) const { return ((float*)_ddAttrs[MAXX])[i]; }
		__device__ float getmaxy(int i) const { return ((float*)_ddAttrs[MAXY])[i]; }
		__device__ float getmaxz(int i) const { return ((float*)_ddAttrs[MAXZ])[i]; }

		__device__ float volume(int i) const {
			return (getmaxx(i) - getminx(i)) * (getmaxy(i) - getminy(i)) * (getmaxz(i) - getminz(i));
		}

		__device__ void setBV(int i, const BvhBvCompletePort& bvs, int j) {
			minx(i) = bvs.getminx(j), miny(i) = bvs.getminy(j), minz(i) = bvs.getminz(j);
			maxx(i) = bvs.getmaxx(j), maxy(i) = bvs.getmaxy(j), maxz(i) = bvs.getmaxz(j);
		}

		__device__ void setBV(int i, const BOX& bv) {
			minx(i) = bv._min.x, miny(i) = bv._min.y, minz(i) = bv._min.z;
			maxx(i) = bv._max.x, maxy(i) = bv._max.y, maxz(i) = bv._max.z;
		}
		__device__ BOX getBV(int i) const {
			return BOX{ getminx(i), getminy(i), getminz(i), getmaxx(i), getmaxy(i), getmaxz(i) };
		}
		__device__ bool overlaps(int i, const BOX&b) const {
			if (b._min.x >getmaxx(i) || b._max.x < getminx(i)) return false;
			if (b._min.y >getmaxy(i) || b._max.y < getminy(i)) return false;
			if (b._min.z >getmaxz(i) || b._max.z < getminz(i)) return false;
			return true;
		}
		__device__ bool overlaps(int i, int j) const {
			if (getminx(j) >getmaxx(i) || getmaxx(j) < getminx(i)) return false;
			if (getminy(j) >getmaxy(i) || getmaxy(j) < getminy(i)) return false;
			if (getminz(j) >getmaxz(i) || getmaxz(j) < getminz(i)) return false;
			return true;
		}
		__device__ bool contains(int i, const vec3f &v) const  {
			return v.x <= getmaxx(i) && v.x >= getminx(i) &&
				v.y <= getmaxy(i) && v.y >= getminy(i) &&
				v.z <= getmaxz(i) && v.z >= getminz(i);
		}
		/// examine the max corner cover situation
		__device__  unsigned int examine_overlap(int i, const BOX &b) const  {
			unsigned int mark = getmaxx(i) > b._max.x | ((getmaxy(i) > b._max.y) << 1) | ((getmaxz(i) > b._max.z) << 2);
			return mark;
		}
	private:
		enum { MINX, MINY, MINZ, MAXX, MAXY, MAXZ };
};

__global__ void gatherBVs(int size, const int* gatherPos, BvhBvCompletePort from, BvhBvCompletePort to);
__global__ void scatterBVs(int size, const int* scatterPos, BvhBvCompletePort from, BvhBvCompletePort to);

#endif // __BVH_BV_H_