using System;
using TensorFlow;

class TensorFlowIMP
{
    static void Main(string[] args)
    {
        int n = 2048 * 2048 * 2048; // total number of data points

        // create input data tensor
        var inputData = new float[n];
        for (int i = 0; i < n; i++) {
            inputData[i] = i;
        }
        var inputTensor = TensorFlow.Tensor.FromBuffer<float>(new long[] { n }, inputData);

        // create TensorFlow graph and session
        using (var graph = new TFGraph())
        using (var session = new TFSession(graph)) {
            // create TensorFlow operations
            var input = graph.Placeholder(TFDataType.Float);
            var output = graph.Multiply(input, graph.Const(2.0f));

            // run TensorFlow session and fetch output
            var outputTensor = session.Run(new[] { input }, new[] { inputTensor }, new[] { output })[0];

            // extract output data from tensor and print results
            var outputData = outputTensor.ToArray<float>();
            Console.WriteLine("TensorFlow output data:");
            for (int i = 0; i < n; i++) {
                Console.WriteLine(outputData[i]);
            }
        }
    }
}