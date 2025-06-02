#ifndef __BV_H_
#define __BV_H_

#include "origin.h"
class AABB {
public:
    vec3f _min, _max;
public:
    inline __host__ __device__ AABB() {
        _max = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        _min = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
    };
    __host__ __device__ AABB(const AABB& b)  { _min = b._min; _max = b._max; }
    __host__ __device__ AABB(AABB&& b)  { _min = b._min; _max = b._max; }
    __host__ __device__ AABB(const float &minx, const float &miny, const float &minz,
        const float &maxx, const float &maxy, const float &maxz) {
        _min = vec3f(minx, miny, minz);
        _max = vec3f(maxx, maxy, maxz);
    }
    __host__ __device__ AABB(const vec3f &v) { _min = _max = v; }
    __host__ __device__ AABB(const vec3f &v1, const vec3f &v2) {
        _min = vec3f(::fmin(v1.x, v2.x), ::fmin(v1.y, v2.y), ::fmin(v1.z, v2.z));
        _max = vec3f(::fmax(v1.x, v2.x), ::fmax(v1.y, v2.y), ::fmax(v1.z, v2.z));
    }
    __host__ __device__  void combines(const vec3f &b) {
        _min = vec3f(::fmin(_min.x, b.x), ::fmin(_min.y, b.y), ::fmin(_min.z, b.z));
        _max = vec3f(::fmax(_max.x, b.x), ::fmax(_max.y, b.y), ::fmax(_max.z, b.z));
    }
    __host__ __device__  void combines(const float x, const float y, const float z) {
        _min = vec3f(::fmin(_min.x, x), ::fmin(_min.y, y), ::fmin(_min.z, z));
        _max = vec3f(::fmax(_max.x, x), ::fmax(_max.y, y), ::fmax(_max.z, z));
    }
    __host__ __device__  void combines(const AABB &b) {
        _min = vec3f(::fmin(_min.x, b._min.x), ::fmin(_min.y, b._min.y), ::fmin(_min.z, b._min.z));
        _max = vec3f(::fmax(_max.x, b._max.x), ::fmax(_max.y, b._max.y), ::fmax(_max.z, b._max.z));
    }

    __host__ __device__  bool overlaps(const AABB &b) const  {
        if (b._min.x > _max.x || b._max.x < _min.x) return false;
        if (b._min.y > _max.y || b._max.y < _min.y) return false;
        if (b._min.z > _max.z || b._max.z < _min.z) return false;
        return true;
    }
    __host__ __device__ bool contains(const vec3f &v) const  {
        return v.x <= _max.x && v.x >= _min.x &&
            v.y <= _max.y && v.y >= _min.y &&
            v.z <= _max.z && v.z >= _min.z;
    }
    __host__ __device__ float merges(const AABB &a, const AABB &b, float *qualityMetric) {
        _min = vec3f(::fmin(a._min.x, b._min.x), ::fmin(a._min.y, b._min.y), ::fmin(a._min.z, b._min.z));
        _max = vec3f(::fmax(a._max.x, b._max.x), ::fmax(a._max.y, b._max.y), ::fmax(a._max.z, b._max.z));
        *qualityMetric = (a.volume() + b.volume()) / volume();
        return *qualityMetric;
    }
    __host__ __device__ void merges(const AABB &a, const AABB &b) {
        _min = vec3f(::fmin(a._min.x, b._min.x), ::fmin(a._min.y, b._min.y), ::fmin(a._min.z, b._min.z));
        _max = vec3f(::fmax(a._max.x, b._max.x), ::fmax(a._max.y, b._max.y), ::fmax(a._max.z, b._max.z));
    }

    __host__ __device__  float width()  const  { return _max.x - _min.x; }
    __host__ __device__  float height() const  { return _max.y - _min.y; }
    __host__ __device__  float depth()  const  { return _max.z - _min.z; }
    __host__ __device__ auto center() const  -> decltype(vec3f(0, 0, 0)){
        return vec3f((_min.x + _max.x)*0.5, (_min.y + _max.y) *0.5, (_min.z + _max.z)*0.5);
    }
    __host__ __device__  float volume() const  { return width()*height()*depth(); }

    inline __host__ __device__  void empty(){
		_max = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		_min = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
	}
    __host__ __device__ void operator=(const AABB& aabb) {
        _min = aabb._min;
        _max = aabb._max;
    }
};

struct __align__(64) bvhNode {
    int par;
    int lc;
    int rc;
    unsigned int mark;

    AABB bounds[2];
};


using BOX = AABB;

#endif // __BV_H_