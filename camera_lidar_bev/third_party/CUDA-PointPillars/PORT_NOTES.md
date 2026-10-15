# Vendored native inference core

Source: https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars, commit `ce7e2bd694c90207435c8751d61cdb38d48a9f4c`.
The source file copyright/license headers are preserved. The repository LICENSE is Apache-2.0; several inference source files have MIT headers. `NOTICE` identifies the OpenPCDet-derived export tools. The original `README.md` is retained as upstream reference, not as instructions for this project.

This project builds only the native `src/pointpillar` core and `src/common/tensorrt.cpp`. It does not install OpenPCDet, Docker, Python training frameworks, or the upstream tool scripts.

Local changes:

- Replace removed TensorRT binding APIs with tensor-name APIs and `enqueueV3`; preserve execution input/output names explicitly.
- Compile with CUDA 13, C++17, and the Thor `sm_110` target.
- Require the bundled model's fixed engine I/O shapes and dtypes before inference.
- Bound the voxel input to the actual ONNX capacity of 10,000 pillars. Safely drop excess pillars and expose the raw/accepted/truncated counts in the public detector statistics.
- Bound NMS to the highest-scoring 4,096 candidates, consistent with the model's KITTI configuration. Use separate bounded device/host masks instead of allocating an all-anchor 13 GB pinned mask, and handle zero candidates without launching an empty CUDA grid.
- Correct the scatter plugin serialization size to the two serialized `size_t` fields.

The bundled model is trained for KITTI Car, Pedestrian, and Cyclist classes, with forward range `[0,69.12)`, lateral range `[-39.68,39.68)`, and height range `[-3,1)` meters. Inputs use x forward, y left, z up. Output z is the box center. The upstream `BoundingBox.w` field is local x extent (car length); `.l` is local y extent (car width). The public `Box3D` API uses unambiguous `size_x`, `size_y`, `size_z` fields.

The Ouster reflectivity-to-KITTI intensity mapping is provisional. Learned full box dimensions and yaw are predictions; their metric accuracy and detection coverage on this rig are not established by the reference or latency checks.
