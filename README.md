# collision-detection-bvh-GPU
抄了一遍王鑫磊师兄的代码，只写了float简单版本的，用template模板函数我怕我cpu烧了。

### lbvh
AABB的最大值和最小值一共6个float，MINX, MINY, MINZ, MAXX, MAXY, MAXZ ，可以用SOA来存储而不是AOS，这样可以加速。不过我为了写起来简单方便没有这么做。

lbvh建树过程参考下面的文章

https://diglib.eg.org/server/api/core/bitstreams/ad092db2-6aec-4f2c-941d-8687de258f00/content

### 优化记录

1. 最大的包围盒bv不要从cpu算，不然需要将场景包围盒从cpu传到gpu，直接gpu上面开shared memory算。