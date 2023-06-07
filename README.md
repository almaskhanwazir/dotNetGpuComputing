# dotNetGpuComputing



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
Write CUDA and ILGPU kernels in the respective projects:
