using System;

public class SimpleNeuralNetwork
{
    private readonly int inputCount;
    private readonly int hiddenCount;
    private readonly float[,] weightsInputHidden;
    private readonly float[] biasHidden;
    private readonly float[] weightsHiddenOutput;
    private float biasOutput;
    private readonly Random random = new Random(42);

    public int InputCount => inputCount;
    public int HiddenCount => hiddenCount;

    public SimpleNeuralNetwork(int inputs, int hidden)
    {
        if (inputs <= 0 || hidden <= 0)
            throw new ArgumentException("Neuronenzahlen muessen positiv sein.");
        inputCount = inputs;
        hiddenCount = hidden;
        weightsInputHidden = new float[inputs, hidden];
        biasHidden = new float[hidden];
        weightsHiddenOutput = new float[hidden];
        for (int i = 0; i < inputs; i++)
            for (int j = 0; j < hidden; j++)
                weightsInputHidden[i, j] = RandomWeight();
        for (int j = 0; j < hidden; j++)
            weightsHiddenOutput[j] = RandomWeight();
    }

    private float RandomWeight() => (float)(random.NextDouble() * 2.0 - 1.0);
    private static float Sigmoid(float x) => 1f / (1f + (float)Math.Exp(-x));

    private void ValidateInputs(float[] inputs)
    {
        if (inputs == null || inputs.Length != inputCount)
            throw new ArgumentException($"Es werden genau {inputCount} Eingaben benoetigt.");
    }

    public float[] GetHiddenActivations(float[] inputs)
    {
        ValidateInputs(inputs);
        float[] hidden = new float[hiddenCount];
        for (int j = 0; j < hiddenCount; j++)
        {
            float sum = biasHidden[j];
            for (int i = 0; i < inputCount; i++)
                sum += inputs[i] * weightsInputHidden[i, j];
            hidden[j] = Sigmoid(sum);
        }
        return hidden;
    }

    public float GetInputHiddenWeight(int input, int hidden) => weightsInputHidden[input, hidden];
    public float GetHiddenOutputWeight(int hidden) => weightsHiddenOutput[hidden];

    public float Predict(float[] inputs)
    {
        float[] hidden = GetHiddenActivations(inputs);
        float sum = biasOutput;
        for (int j = 0; j < hiddenCount; j++)
            sum += hidden[j] * weightsHiddenOutput[j];
        return Sigmoid(sum);
    }

    public float Train(float[] inputs, float target, float learningRate)
    {
        ValidateInputs(inputs);
        if (target < 0f || target > 1f)
            throw new ArgumentOutOfRangeException(nameof(target));
        if (learningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(learningRate));

        float[] hidden = GetHiddenActivations(inputs);
        float sum = biasOutput;
        for (int j = 0; j < hiddenCount; j++)
            sum += hidden[j] * weightsHiddenOutput[j];
        float output = Sigmoid(sum);
        float error = target - output;
        float outputDelta = error * output * (1f - output);
        float[] hiddenDeltas = new float[hiddenCount];
        for (int j = 0; j < hiddenCount; j++)
            hiddenDeltas[j] = outputDelta * weightsHiddenOutput[j] * hidden[j] * (1f - hidden[j]);
        for (int j = 0; j < hiddenCount; j++)
            weightsHiddenOutput[j] += learningRate * outputDelta * hidden[j];
        biasOutput += learningRate * outputDelta;
        for (int i = 0; i < inputCount; i++)
            for (int j = 0; j < hiddenCount; j++)
                weightsInputHidden[i, j] += learningRate * hiddenDeltas[j] * inputs[i];
        for (int j = 0; j < hiddenCount; j++)
            biasHidden[j] += learningRate * hiddenDeltas[j];
        return error * error;
    }
}
