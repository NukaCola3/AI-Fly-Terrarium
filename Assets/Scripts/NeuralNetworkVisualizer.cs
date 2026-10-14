using UnityEngine;

public class NeuralNetworkVisualizer : MonoBehaviour
{
    [Header("Training")]
    [SerializeField, Min(1)] private int maxEpochs = 10000;
    [SerializeField, Min(1)] private int epochsPerFrame = 10;
    [SerializeField, Min(0.0001f)] private float learningRate = 0.5f;
    [SerializeField] private bool trainAutomatically = true;

    [Header("Darstellung")]
    [SerializeField] private Vector2 nodeSize = new Vector2(58, 36);
    [SerializeField] private bool showWeights = true;

    private SimpleNeuralNetwork network;
    private int epoch;
    private float meanError;
    private int selectedExample;
    private readonly float[][] examples =
    {
        new float[] {1f, 1f, 1f},
        new float[] {0.1f, 1f, 0.2f},
        new float[] {0.1f, 0.2f, 1f},
        new float[] {0.2f, 1f, 1f},
        new float[] {1f, 0.1f, 1f},
        new float[] {1f, 1f, 0.1f}
    };
    private readonly float[] targets = {0.5f, 0f, 1f, 0.5f, 1f, 0f};
    private readonly string[] exampleNames =
    {
        "Freie Strecke", "Vorne blockiert, links frei",
        "Vorne blockiert, rechts frei", "Vorne nah, beide Seiten frei",
        "Wand links", "Wand rechts"
    };
    private Texture2D whiteTexture;
    private GUIStyle centerStyle;
    private GUIStyle smallStyle;

    private void Awake()
    {
        network = new SimpleNeuralNetwork(3, 5);
        whiteTexture = Texture2D.whiteTexture;

    }

    private void Update()
    {
        if (!trainAutomatically || epoch >= maxEpochs) return;
        int count = Mathf.Min(epochsPerFrame, maxEpochs - epoch);
        for (int e = 0; e < count; e++)
        {
            float error = 0f;
            for (int i = 0; i < examples.Length; i++)
                error += network.Train(examples[i], targets[i], learningRate);
            meanError = error / examples.Length;
            epoch++;
        }
    }

    private void OnGUI()
    {
        if (centerStyle == null)
        {
            centerStyle = new GUIStyle(GUI.skin.label)
            {
                alignment = TextAnchor.MiddleCenter,
                fontSize = 13,
                normal = { textColor = Color.white }
            };
            smallStyle = new GUIStyle(centerStyle) { fontSize = 11 };
        }
        if (network == null) return;
        float width = Mathf.Min(Screen.width - 20f, 950f);
        float height = Mathf.Min(Screen.height - 20f, 660f);
        Rect panel = new Rect(10f, 10f, width, height);
        GUI.Box(panel, "Neuronales Netz - Live-Training");

        GUILayout.BeginArea(new Rect(25f, 40f, width - 30f, 125f));
        GUILayout.Label($"Epoche: {epoch} / {maxEpochs}    Mittlerer Fehler: {meanError:F6}");
        GUILayout.BeginHorizontal();
        if (GUILayout.Button(trainAutomatically ? "Pause" : "Weiter", GUILayout.Width(90)))
            trainAutomatically = !trainAutomatically;
        if (GUILayout.Button("Neu starten", GUILayout.Width(110)))
        {
            network = new SimpleNeuralNetwork(3, 5);
            epoch = 0;
            meanError = 0f;
            trainAutomatically = true;
        }
        GUILayout.EndHorizontal();
        GUILayout.BeginHorizontal();
        GUILayout.Label("Beispiel:", GUILayout.Width(65));
        if (GUILayout.Button("<", GUILayout.Width(32)))
            selectedExample = (selectedExample + examples.Length - 1) % examples.Length;
        GUILayout.Label(exampleNames[selectedExample], GUILayout.Width(260));
        if (GUILayout.Button(">", GUILayout.Width(32)))
            selectedExample = (selectedExample + 1) % examples.Length;
        GUILayout.EndHorizontal();
        GUILayout.EndArea();

        float[] inputs = examples[selectedExample];
        float[] hidden = network.GetHiddenActivations(inputs);
        float output = network.Predict(inputs);
        float top = 210f;
        float bottom = Mathf.Max(top + 100f, height - 105f);
        float x1 = 105f;
        float x2 = width * 0.50f;
        float x3 = width - 115f;
        Vector2[] inputPoints = MakePoints(x1, 3, top, bottom);
        Vector2[] hiddenPoints = MakePoints(x2, 5, top, bottom);
        Vector2[] outputPoints = MakePoints(x3, 1, top, bottom);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 5; j++)
                DrawConnection(inputPoints[i], hiddenPoints[j],
                    network.GetInputHiddenWeight(i, j));
        for (int j = 0; j < 5; j++)
            DrawConnection(hiddenPoints[j], outputPoints[0],
                network.GetHiddenOutputWeight(j));

        for (int i = 0; i < 3; i++)
            DrawNode(inputPoints[i], inputs[i], $"I{i + 1}");
        for (int j = 0; j < 5; j++)
            DrawNode(hiddenPoints[j], hidden[j], $"H{j + 1}");
        DrawNode(outputPoints[0], output, "OUT");

        GUI.Label(new Rect(x1 - 65, top - 55, 130, 25), "Eingaben", centerStyle);
        GUI.Label(new Rect(x2 - 85, top - 55, 170, 25), "Hidden Layer", centerStyle);
        GUI.Label(new Rect(x3 - 75, top - 55, 150, 25), "Ausgabe", centerStyle);
        GUI.Label(new Rect(30, height - 55, width - 50, 30),
            $"Ausgabe: {output:F4}  |  Ziel: {targets[selectedExample]:F2}  |  Blau: positives Gewicht  /  Rot: negatives Gewicht",
            smallStyle);
    }

    private Vector2[] MakePoints(float x, int count, float top, float bottom)
    {
        Vector2[] points = new Vector2[count];
        for (int i = 0; i < count; i++)
            points[i] = new Vector2(x, count == 1 ? (top + bottom) * 0.5f :
                Mathf.Lerp(top, bottom, (float)i / (count - 1)));
        return points;
    }

    private void DrawNode(Vector2 p, float value, string label)
    {
        Rect rect = new Rect(p.x - nodeSize.x * 0.5f,
            p.y - nodeSize.y * 0.5f, nodeSize.x, nodeSize.y);
        Color old = GUI.color;
        GUI.color = Color.Lerp(new Color(0.16f, 0.20f, 0.28f),
            new Color(0.1f, 0.75f, 0.4f), Mathf.Clamp01(value));
        GUI.DrawTexture(rect, whiteTexture);
        GUI.color = Color.white;
        GUI.Label(rect, $"{label}\n{value:F2}", smallStyle);
        GUI.color = old;
    }

    private void DrawConnection(Vector2 a, Vector2 b, float weight)
    {
        Color old = GUI.color;
        GUI.color = weight >= 0f
            ? new Color(0.25f, 0.65f, 1f, 0.85f)
            : new Color(1f, 0.35f, 0.35f, 0.85f);
        Vector2 delta = b - a;
        float angle = Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg;
        Matrix4x4 previous = GUI.matrix;
        GUIUtility.RotateAroundPivot(angle, a);
        GUI.DrawTexture(new Rect(a.x, a.y - Mathf.Clamp(Mathf.Abs(weight), 1f, 4f) * 0.5f,
            delta.magnitude, Mathf.Clamp(Mathf.Abs(weight), 1f, 4f)), whiteTexture);
        GUI.matrix = previous;
        GUI.color = old;
        if (showWeights)
        {
            Vector2 mid = (a + b) * 0.5f;
            GUI.Label(new Rect(mid.x - 23, mid.y - 10, 46, 20), weight.ToString("F2"), smallStyle);
        }
    }
}
