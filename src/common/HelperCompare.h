#ifndef IVCOLLISION_COMMON_HELPERCOMPARE_H
#define IVCOLLISION_COMMON_HELPERCOMPARE_H

#include "Common.h"

#include <iostream>
#include <set>

////////////////////////////////////////////////////////////////////////////////
// equal
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ bool operator==(const int3 &a, const int3 &b)
{
	return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}
inline __host__ __device__ bool operator==(const int4 &a, const int4 &b)
{
	return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);
}
inline __host__ __device__ bool operator!=(const int4 &a, const int4 &b)
{
	return !(a == b);
}
inline __host__ __device__ bool nearlyEqual(float a, float b)
{
	// if (a == b) return true;
	// float diff = fabs(a - b);
	// float norm = fminf(fabs(a) + fabs(b), FLT_MAX);
	// return diff < fmaxf(FLT_MIN, (FLT_EPSILON * 128.0f) * norm);
	return fabs(a - b) <= FLT_EPSILON * 16.0f;
}
inline __host__ __device__ bool operator==(const float3 &a, const float3 &b)
{
	return nearlyEqual(a.x, b.x) && nearlyEqual(a.y, b.y) && nearlyEqual(a.z, b.z);
}

////////////////////////////////////////////////////////////////////////////////
// less
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ bool operator<(const int3 &a, const int3 &b)
{
	if (a.x != b.x)
		return a.x < b.x;
	if (a.y != b.y)
		return a.y < b.y;
	if (a.z != b.z)
		return a.z < b.z;
	return false;
}
inline __host__ __device__ bool operator<(const int4 &a, const int4 &b)
{
	if (a.x != b.x)
		return a.x < b.x;
	if (a.y != b.y)
		return a.y < b.y;
	if (a.z != b.z)
		return a.z < b.z;
	if (a.w != b.w)
		return a.w < b.w;
	return false;
}

namespace IVCollision
{
	//================================================================================
	// Print Info Functions
	//================================================================================

	inline std::ostream &operator<<(std::ostream &_os, const int4 &_p)
	{
		_os << "(" << _p.x << "," << _p.y << "," << _p.z << "," << _p.w << ")";
		return _os;
	}

	inline std::ostream &operator<<(std::ostream &_os, const float3 &_p)
	{
		_os << "(" << _p.x << "," << _p.y << "," << _p.z << ")";
		return _os;
	}

	inline void FuncPrintFailedInfo(int diff, std::string variable, std::string function)
	{
		printf("[\x1b[31mERROR\x1b[0m] Test Failed => { %s } %d differences found. (Function: %s).\n",
			   variable.c_str(), diff, function.c_str());
	}

	inline void FuncPrintSuccessInfo(int diff, std::string variable, std::string function)
	{
		printf("[\x1b[32mINFO\x1b[0m] Test Pass => { %s } (Function: %s).\n",
			   variable.c_str(), function.c_str());
	}

	inline void FuncPrintDiffInfo(int diff, std::string variable, std::string function)
	{
		if (diff > 0)
		{
			FuncPrintFailedInfo(diff, variable, function);
		}
		else
		{
			FuncPrintSuccessInfo(diff, variable, function);
		}
	}

	//================================================================================
	// Comparators
	//================================================================================

	template <typename T>
	int CompareVector(const std::vector<T> &v1, const std::vector<T> &v2,
					  bool verbose = false)
	{
		int diffs = 0;
		for (int i = 0; i < max(int(v1.size()), int(v2.size())); i++)
		{
			if (i >= v1.size())
			{
				diffs++;
				if (verbose)
				{
					std::cout << " | -------- v2[" << i << "] = " << v2[i] << " not in v1" << std::endl;
				}
				continue;
			}
			if (i >= v2.size())
			{
				diffs++;
				if (verbose)
				{
					std::cout << " | -------- v1[" << i << "] = " << v1[i] << " not in v2" << std::endl;
				}
				continue;
			}
			if (!(v1[i] == v2[i]))
			{
				diffs++;
				if (verbose)
				{
					std::cout << " | -------- v1[" << i << "] = " << v1[i] << ", v2[" << i << "] = " << v2[i] << ", diff = " << v1[i] - v2[i] << std::endl;
				}
			}
		}
		return diffs;
	}

	template <typename T>
	int CompareSortedVectorAsSet(const std::vector<T> &v1, const std::vector<T> &v2,
								 bool verbose = false)
	{
		std::set<T> s1, s2;
		s1.insert(v1.begin(), v1.end());
		s2.insert(v2.begin(), v2.end());
		int diffs = 0;
		for (int i = 0; i < max(int(v1.size()), int(v2.size())); i++)
		{
			if (i < v2.size() && (s1.find(v2[i]) == s1.end()))
			{
				diffs++;
				if (verbose)
				{
					std::cout << " | -------- v2[" << i << "] = " << v2[i] << " not in v1" << std::endl;
				}
			}
			if (i < v1.size() && (s2.find(v1[i]) == s2.end()))
			{
				diffs++;
				if (verbose)
				{
					std::cout << " | -------- v1[" << i << "] = " << v1[i] << " not in v2" << std::endl;
				}
			}
		}
		return diffs;
	}

	template <typename T>
	int CompareVectorVector(const std::vector<std::vector<T>> &v1, const std::vector<std::vector<T>> &v2,
							bool verbose = false,
							bool compareAsSet = true)
	{
		int diffs = 0;
		for (int i = 0; i < max(int(v1.size()), int(v2.size())); i++)
		{
			if (i >= v1.size())
			{
				diffs += static_cast<int>(v2[i].size());
				continue;
			}
			if (i >= v2.size())
			{
				diffs += static_cast<int>(v1[i].size());
				continue;
			}
			int diffv = compareAsSet ? CompareSortedVectorAsSet(v1[i], v2[i], verbose)
									 : CompareVector(v1[i], v2[i]);
			if (verbose && diffv > 0)
			{
				printf(" | ---- Diff result between v1[%d] and v2[%d]\n", i, i);
			}

			diffs += diffv;
		}
		return diffs;
	}
}

#endif
