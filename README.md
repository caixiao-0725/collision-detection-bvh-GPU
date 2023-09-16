# collision-detection-bvh-GPU
抄了一遍王鑫磊师兄的代码，只写了float简单版本的，用template模板函数我怕我cpu烧了。

### lbvh
AABB的最大值和最小值一共6个float，MINX, MINY, MINZ, MAXX, MAXY, MAXZ ，可以用SOA来存储而不是AOS，这样可以加速。

树的结构也要用SOA来存储。