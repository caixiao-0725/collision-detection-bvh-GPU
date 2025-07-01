# collision-detection-bvh-GPU

参考王鑫磊师兄的代码，代码有点老了，现在的机器上跑会出错，我的机子是一台3080。

### lbvh

lbvh建树过程参考下面的文章

https://diglib.eg.org/server/api/core/bitstreams/ad092db2-6aec-4f2c-941d-8687de258f00/content

参考仓库：

https://github.com/ZiXuanVickyLu/culbvh

https://github.com/littlemine/BVH-based-Collision-Detection-Scheme

https://github.com/jerry060599/KittenGpuLBVH

### 纠错记录

在建树的过程中需要加入threadfence，因为一个线程向global memory 中写入内存仅该线程可见，别的线程会从缓存中读取内存，所以有些向global memory中写的操作会被忽略，导致建树错误。

### 优化记录

1. 最大的包围盒bv不要从cpu算，不然需要将场景包围盒从cpu传到gpu，直接gpu上面开shared memory算

2. SOA < AOS

3. 去掉了d_keys32，直接用_extNodes._mtcode。这里面ext和int的意思就是，ext是叶子节点，int是除叶子节点以外的节点，morton code和叶子节点一一对应，减少一次copy

4. 不用thrust库，而使用cub库，因为thrust库在sort的时候会malloc和free内存(这个要不就别写了，cub调用起来又长又丑，不方便食用)

5. 去掉mark，在师兄的代码里面，mark的最后三位分别标记了该节点的左子节点，右子节点，还有它本身是否是叶子节点，而这个mark可以分别隐藏在节点的索引里面

6. 可以对查询的元素进行排序，在自碰撞里面有用，因为自碰撞的元素在建树的时候已经排过序了，直接把索引拿过来用，如果是和其他物体的碰撞，不建议排序，因为排序很消耗时间。代码里11和12进行了对比，146us：48us，感觉这个加速还是挺惊人的。但是我用的场景只有356个碰撞，所以在写入内存这一块没有考虑进去

7. stackless query 创建了32bytes的结构体，(左孩子索引，escape索引，aabb包围盒共32字节)，在遍历的时候可以合并访问，type = 4

### optix collision detection

使用optix去做碰撞检测，基础单元使用三角形的话有诸多问题，弃用，但是可以参考我的某一次commit，它无法处理光线穿过两个三角形的公共边，并且无法处理有厚度的情况。采用光线和aabb包围盒，自己做mid phase。

在使用optix的过程中，发现optix会出现一些false positive的情况，请教newton发现optix用的是half精度，所以包围盒会比float的大一点。

在用optix建树的时候，发现建树有点慢，是自己的写的4倍慢，有可能是我的建树的某些参数没设置。但是optix的query是比传统的方法要快的。

### 比optix更快更准的方法！

基于stackless query，想办法将32字节压缩至16字节，时间直接缩短了一半，比optix还快还准哦。(type = 5)

压缩方法：申请一个long long int，64个bit位，前面19位用来存储索引，后面15*3位用来存aabb的三个float，这样就可以用两个long long int存储一个树的节点啦。并且英伟达的float4可以合批访问，只用一个指令就可以取得数据。

19位索引只能存2^19个数，可以调整成22位，这样就能处理百万量级的碰撞，但是相应的精度就低了，会有更多的false positive。

对比half精度的做法，half的尾数有10位，但是我的15位全部用来存储尾数(基数和正负符号位都可以扔了)，所以我的false positive的数量是少于half的。

最终结果比opitx快了10%。