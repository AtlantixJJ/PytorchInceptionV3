# Pytorch InceptionV3 v.s. Tensorflow Inception V3

run: `python test_inception.py`.

Expected output:

```
/data/home/xujianjing/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
(10, 2048)
[0.08839414 0.09941835 0.26680773 ... 0.17973183 0.16615462 0.8939992 ]
WARNING:tensorflow:From test_inception.py:65: FastGFile.__init__ (from tensorflow.python.platform.gfile) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.gfile.GFile.
2019-05-22 10:16:44.477456: W tensorflow/core/framework/op_def_util.cc:355] Op BatchNormWithGlobalNormalization is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
2019-05-22 10:16:44.644898: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-05-22 10:16:44.654593: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2299935000 Hz
2019-05-22 10:16:44.657962: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x563ba68d8910 executing computations on platform Host. Devices:
2019-05-22 10:16:44.658013: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-05-22 10:16:44.662613: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x563baf2bdc50 executing computations on platform CUDA. Devices:
2019-05-22 10:16:44.662657: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): GeForce GTX TITAN X, Compute Capability 5.2
2019-05-22 10:16:44.663293: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX TITAN X major: 5 minor: 2 memoryClockRate(GHz): 1.076
pciBusID: 0000:8a:00.0
totalMemory: 11.92GiB freeMemory: 1.20GiB
2019-05-22 10:16:44.663350: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-05-22 10:16:44.663387: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-05-22 10:16:44.663406: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-05-22 10:16:44.663422: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-05-22 10:16:44.663730: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1008 MB memory) -> physical GPU (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:8a:00.0, compute capability: 5.2)
2019-05-22 10:16:46.949924: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 678.75MiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-05-22 10:16:47.033009: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.17GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-05-22 10:16:47.048306: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.9.0 locally
2019-05-22 10:16:47.063510: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.00GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-05-22 10:16:47.133744: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 514.83MiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2019-05-22 10:16:47.231971: W tensorflow/core/common_runtime/bfc_allocator.cc:211] Allocator (GPU_0_bfc) ran out of memory trying to allocate 748.55MiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
=> Pytorch pool3:
[0.08839414 0.09941835 0.26680773 0.22294888 0.00075007 0.2637058 ]
=> Tensorflow pool3:
[0.48264152 0.15095681 0.27383083 0.08546548 0.39348015 0.12462334]
=> Mean abs difference
0.34575087
```

`pool3` is the feature used for Frechet Inception Distance. Pytorch feature differs with Tensorflow feature.