
using UnityEngine;

public class NeuralNetworkTest : MonoBehaviour
{
    [Header("Training")]
    [SerializeField] private int epochs = 10000;
    [SerializeField] private float learningRate = 0.5f;

    private SimpleNeuralNetwork network;

    private void Start()
    {
        network = new SimpleNeuralNetwork(3, 5);

        // Eingaben: vorne, links, rechts
        // Kleine Werte = Wand ist nah
        float[][] trainingInputs =
        {
            new float[] { 1f, 1f, 1f },
            new float[] { 0.1f, 1f, 0.2f },
            new float[] { 0.1f, 0.2f, 1f },
            new float[] { 0.2f, 1f, 1f },
            new float[] { 1f, 0.1f, 1f },
            new float[] { 1f, 1f, 0.1f }
        };

        // 0 = links, 0.5 = geradeaus, 1 = rechts
        float[] targets =
        {
            0.5f,
            0f,
            1f,
            0.5f,
            1f,
            0f
        };

        Debug.Log("Training gestartet");

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float totalError = 0f;

            for (int i = 0; i < trainingInputs.Length; i++)
            {
                totalError += network.Train(
                    trainingInputs[i],
                    targets[i],
                    learningRate
                );
            }

            if ((epoch + 1) % 1000 == 0)
            {
                Debug.Log(
                    $"Epoche {epoch + 1} | " +
                    $"Mittlerer Fehler: " +
                    $"{totalError / trainingInputs.Length:F6}"
                );
            }
        }

        Debug.Log("Training abgeschlossen");

        TestNetwork("Freie Strecke",
            new float[] { 1f, 1f, 1f });

        TestNetwork("Wand links",
            new float[] { 1f, 0.1f, 1f });

        TestNetwork("Wand rechts",
            new float[] { 1f, 1f, 0.1f });

        TestNetwork("Hindernis vorne, links frei",
            new float[] { 0.1f, 1f, 0.2f });

        TestNetwork("Hindernis vorne, rechts frei",
            new float[] { 0.1f, 0.2f, 1f });
    }

    private void TestNetwork(string description, float[] inputs)
    {
        float result = network.Predict(inputs);

        Debug.Log(
            $"{description} | Netzwerkausgabe: {result:F4}"
        );
    }
}
