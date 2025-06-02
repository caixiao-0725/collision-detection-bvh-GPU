# collision-detection-bvh-GPU

参考王鑫磊的代码，代码有点老了，现在的机器上跑会出错。

### lbvh

lbvh建树过程参考下面的文章

https://diglib.eg.org/server/api/core/bitstreams/ad092db2-6aec-4f2c-941d-8687de258f00/content

### 纠错记录

在建树的过程中需要加入threadfence，因为一个线程向global memory 中写入内存仅该线程可见，别的线程会从缓存中读取内存，所以有些向global memory中写的操作会被忽略，导致建树错误。

### 优化记录

1. 最大的包围盒bv不要从cpu算，不然需要将场景包围盒从cpu传到gpu，直接gpu上面开shared memory算

2. SOA < AOS

3. 去掉了d_keys32，直接用_extNodes._mtcode。这里面ext和int的意思就是，ext是叶子节点，int是除叶子节点以外的节点，morton code和叶子节点一一对应，减少一次copy

4. 不用thrust 库，转而使用cub库，因为thrust库在sort的时候会malloc和free内存