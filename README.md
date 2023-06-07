# dotNetGpuComputing

You want to perform a complex calculation and utilize GPU computing to speed up the process. Unfortunately, .NET Core does not have native support for GPU computing. However, you can use external libraries to achieve this.

This solution contains 3 classes:

ManagedCudaSample - Uses the ManagedCuda library to execute CUDA kernels on the GPU.
TensorFlowSample - Uses TensorFlow.NET to build and run TensorFlow models on the GPU.
ILGPUSample - Uses the ILGPU library to compile C# code to GPU kernels.
Steps:

Install these NuGet packages:
ManagedCudaSample.csproj:

ManagedCuda
TensorFlowSample.csproj:

TensorFlow.NET
ILGPUSample.csproj:

ILGPU
Write CUDA and ILGPU kernels in the respective projects




