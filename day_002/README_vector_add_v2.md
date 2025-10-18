On T4 colab
Imp to change arch type, to let compiler know how to optmize code

!nvcc -arch=sm_75

performing warm up runsbenchmarking CPU implementation
benchmarking 1d GPU implementation
1D Results are correct
benchmarking 3d GPU implementation
3D Results are correct
CPU average time: 30.405118 milliseconds
GPU 1D average time: 12.985687 milliseconds
GPU 3D average time: 0.832030 milliseconds
Speedup (CPU vs GPU 1D): 2.341433x
Speedup (CPU vs GPU 3D): 36.543287x
Speedup (GPU 1D vs GPU 3D): 15.607230x
