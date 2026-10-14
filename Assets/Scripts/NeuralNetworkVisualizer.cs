using System.Collections.Generic;
using UnityEngine;

// Dateiinhalte in NeuralNetworkVisualizer.cs kopieren.
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
    [SerializeField, Range(1, 200)] private int historyEveryEpochs = 25;

    private SimpleNeuralNetwork network;
    private int epoch;
    private float meanError;
    private int selectedExample;
    private bool manualSensors;
    private readonly float[] sensorValues = { 1f, 1f, 1f };
    private readonly float[][] examples =
    {
        new float[] {1f, 1f, 1f}, new float[] {0.1f, 1f, 0.2f},
        new float[] {0.1f, 0.2f, 1f}, new float[] {0.2f, 1f, 1f},
        new float[] {1f, 0.1f, 1f}, new float[] {1f, 1f, 0.1f}
    };
    private readonly float[] targets = { 0.5f, 0f, 1f, 0.5f, 1f, 0f };
    private readonly string[] exampleNames =
    {
        "Freie Strecke", "Vorne blockiert, links frei",
        "Vorne blockiert, rechts frei", "Vorne nah, beide Seiten frei",
        "Wand links", "Wand rechts"
    };

    // 15 Eingabe->Hidden + 5 Hidden->Ausgabe = 20 Gewichte.
    private readonly float[] previousWeights = new float[20];
    private readonly float[] weightChanges = new float[20];
    private readonly List<float>[] histories = new List<float>[20];
    private readonly List<int> historyEpochs = new List<int>();
    private int selectedWeight;
    private bool heatmapShowChanges = true;
    private float heatmapScale = 0.01f;
    private Texture2D whiteTexture;
    private GUIStyle centerStyle;
    private GUIStyle smallStyle;

    private void Awake()
    {
        whiteTexture = Texture2D.whiteTexture;
        for (int i = 0; i < histories.Length; i++)
            histories[i] = new List<float>();
        ResetNetwork();
    }

    private float WeightAt(int index)
    {
        if (index < 15)
            return network.GetInputHiddenWeight(index / 5, index % 5);
        return network.GetHiddenOutputWeight(index - 15);
    }

    private string WeightName(int index)
    {
        if (index < 15)
            return $"I{index / 5 + 1} -> H{index % 5 + 1}";
        return $"H{index - 14} -> OUT";
    }

    private void ResetNetwork()
    {
        network = new SimpleNeuralNetwork(3, 5);
        epoch = 0;
        meanError = 0f;
        trainAutomatically = true;
        historyEpochs.Clear();
        for (int i = 0; i < 20; i++)
        {
            histories[i].Clear();
            previousWeights[i] = WeightAt(i);
            weightChanges[i] = 0f;
        }
        RecordHistory();
    }

    private void RecordHistory()
    {
        historyEpochs.Add(epoch);
        for (int i = 0; i < 20; i++)
            histories[i].Add(WeightAt(i));
        // Begrenzt den Speicherbedarf; Verlauf zeigt die letzten 240 Messpunkte.
        if (historyEpochs.Count > 240)
        {
            historyEpochs.RemoveAt(0);
            for (int i = 0; i < 20; i++) histories[i].RemoveAt(0);
        }
    }

    private void TrainEpoch()
    {
        if (epoch >= maxEpochs) return;
        float error = 0f;
        for (int i = 0; i < examples.Length; i++)
            error += network.Train(examples[i], targets[i], learningRate);
        meanError = error / examples.Length;
        epoch++;
        if (epoch % historyEveryEpochs == 0 || epoch == maxEpochs)
            RecordHistory();
    }

    private void Update()
    {
        if (!trainAutomatically || epoch >= maxEpochs) return;
        int count = Mathf.Min(epochsPerFrame, maxEpochs - epoch);
        for (int e = 0; e < count; e++) TrainEpoch();
        RefreshChanges();
    }

    private void RefreshChanges()
    {
        for (int i = 0; i < 20; i++)
        {
            float current = WeightAt(i);
            weightChanges[i] = current - previousWeights[i];
            previousWeights[i] = current;
        }
    }

    private void OnGUI()
    {
        if (centerStyle == null)
        {
            centerStyle = new GUIStyle(GUI.skin.label)
            {
                alignment = TextAnchor.MiddleCenter, fontSize = 13,
                normal = { textColor = Color.white }
            };
            smallStyle = new GUIStyle(centerStyle) { fontSize = 11 };
        }
        if (network == null) return;

        float width = Mathf.Min(Screen.width - 20f, 1050f);
        float height = Mathf.Min(Screen.height - 20f, 1100f);
        GUI.Box(new Rect(10f, 10f, width, height), "Neuronales Netz - Gewichte, Heatmap und Training");
        GUILayout.BeginArea(new Rect(25f, 40f, width - 30f, 205f));
        GUILayout.Label($"Epoche: {epoch} / {maxEpochs}    Mittlerer Fehler: {meanError:F6}");
        GUILayout.BeginHorizontal();
        if (GUILayout.Button(trainAutomatically ? "Pause" : "Weiter", GUILayout.Width(90)))
            trainAutomatically = !trainAutomatically;
        if (GUILayout.Button("1 Epoche", GUILayout.Width(95)))
        {
            trainAutomatically = false;
            TrainEpoch();
            RefreshChanges();
            // Auch Einzelschritte in der Kurve anzeigen.
            if (historyEpochs[historyEpochs.Count - 1] != epoch) RecordHistory();
        }
        if (GUILayout.Button("Neu starten", GUILayout.Width(110))) ResetNetwork();
        GUILayout.EndHorizontal();
        GUILayout.BeginHorizontal();
        GUILayout.Label("Beispiel:", GUILayout.Width(65));
        if (GUILayout.Button("<", GUILayout.Width(32)))
        {
            selectedExample = (selectedExample + examples.Length - 1) % examples.Length;
            manualSensors = false;
        }
        GUILayout.Label(exampleNames[selectedExample], GUILayout.Width(260));
        if (GUILayout.Button(">", GUILayout.Width(32)))
        {
            selectedExample = (selectedExample + 1) % examples.Length;
            manualSensors = false;
        }
        GUILayout.EndHorizontal();
        bool nextManual = GUILayout.Toggle(manualSensors, "Sensoren manuell einstellen");
        if (nextManual && !manualSensors)
            System.Array.Copy(examples[selectedExample], sensorValues, 3);
        manualSensors = nextManual;
        if (manualSensors)
        {
            DrawSensorSlider("Vorne", 0);
            DrawSensorSlider("Links", 1);
            DrawSensorSlider("Rechts", 2);
        }
        else GUILayout.Label("Manuellen Modus aktivieren, um Sensorwerte live zu testen.");
        GUILayout.EndArea();

        // Auswahl eines beliebigen der 20 Gewichte und sein Verlauf.
        float graphTop = 255f;
        GUI.Box(new Rect(25, graphTop, width - 30, 175), "Gewichtsmonitor");
        GUILayout.BeginArea(new Rect(35, graphTop + 25, width - 50, 28));
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("<", GUILayout.Width(32))) selectedWeight = (selectedWeight + 19) % 20;
        GUILayout.Label($"Gewicht {selectedWeight + 1}/20: {WeightName(selectedWeight)}", GUILayout.Width(225));
        if (GUILayout.Button(">", GUILayout.Width(32))) selectedWeight = (selectedWeight + 1) % 20;
        GUILayout.Label($"Wert: {WeightAt(selectedWeight):+0.0000;-0.0000;0.0000}    Aenderung im letzten Schritt: {weightChanges[selectedWeight]:+0.000000;-0.000000;0.000000}");
        GUILayout.EndHorizontal();
        GUILayout.EndArea();
        DrawHistory(new Rect(48, graphTop + 62, width - 80, 100));

        DrawHeatmap(new Rect(25f, 445f, width - 30f, 245f));

        float[] inputs = manualSensors ? sensorValues : examples[selectedExample];
        float[] hidden = network.GetHiddenActivations(inputs);
        float output = network.Predict(inputs);
        float top = 755f;
        float bottom = Mathf.Max(top + 90f, height - 90f);
        float x1 = 105f;
        float x2 = width * 0.50f;
        float x3 = width - 115f;
        Vector2[] inputPoints = MakePoints(x1, 3, top, bottom);
        Vector2[] hiddenPoints = MakePoints(x2, 5, top, bottom);
        Vector2[] outputPoints = MakePoints(x3, 1, top, bottom);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 5; j++)
                DrawConnection(inputPoints[i], hiddenPoints[j], network.GetInputHiddenWeight(i, j), i * 5 + j);
        for (int j = 0; j < 5; j++)
            DrawConnection(hiddenPoints[j], outputPoints[0], network.GetHiddenOutputWeight(j), 15 + j);
        for (int i = 0; i < 3; i++) DrawNode(inputPoints[i], inputs[i], $"I{i + 1}");
        for (int j = 0; j < 5; j++) DrawNode(hiddenPoints[j], hidden[j], $"H{j + 1}");
        DrawNode(outputPoints[0], output, "OUT");
        GUI.Label(new Rect(x1 - 65, top - 48, 130, 25), "Eingaben", centerStyle);
        GUI.Label(new Rect(x2 - 85, top - 48, 170, 25), "Hidden Layer", centerStyle);
        GUI.Label(new Rect(x3 - 75, top - 48, 150, 25), "Ausgabe", centerStyle);
        string decision = output < 0.35f ? "Links fliegen" : output > 0.65f ? "Rechts fliegen" : "Geradeaus";
        string targetInfo = manualSensors ? "Manuelle Eingaben (kein Zielwert)" : $"Ziel: {targets[selectedExample]:F2}";
        GUI.Label(new Rect(30, height - 53, width - 50, 30),
            $"Ausgabe: {output:F4}  |  {decision}  |  {targetInfo}", smallStyle);
    }


    // 3x5 Input->Hidden und 1x5 Hidden->Output: 20 Gewichte.
    // Die Farben zeigen wahlweise Gewichtsveraenderungen oder absolute Gewichte.
    private void DrawHeatmap(Rect area)
    {
        GUI.Box(area, "Live-Heatmap (20 Gewichte)");
        Rect toggle = new Rect(area.x + 12f, area.y + 24f, 240f, 23f);
        heatmapShowChanges = GUI.Toggle(toggle, heatmapShowChanges,
            "Aenderung seit letztem Update");

        float maxAbs = 0.000001f;
        for (int i = 0; i < 20; i++)
        {
            float v = heatmapShowChanges ? weightChanges[i] : WeightAt(i);
            maxAbs = Mathf.Max(maxAbs, Mathf.Abs(v));
        }
        // Gemeinsame Skala fuer alle Zellen, damit Farben vergleichbar bleiben.
        float scale = heatmapShowChanges ? Mathf.Max(0.000001f, maxAbs) : Mathf.Max(1f, maxAbs);
        GUI.Label(new Rect(area.x + 265f, area.y + 25f, area.width - 275f, 22f),
            heatmapShowChanges ? $"Skala: +/-{scale:F6} (pro Update)" : $"Skala: +/-{scale:F3} (Gewicht)", smallStyle);

        float left = area.x + 75f;
        float top = area.y + 70f;
        float cellW = Mathf.Min(90f, (area.width - 120f) / 6f);
        float cellH = 33f;
        float gap = 5f;
        for (int j = 0; j < 5; j++)
        {
            GUI.Label(new Rect(left + j * (cellW + gap), top - 23f, cellW, 20f), $"H{j + 1}", centerStyle);
        }
        for (int i = 0; i < 3; i++)
        {
            GUI.Label(new Rect(area.x + 20f, top + i * (cellH + gap), 45f, cellH), $"I{i + 1}", centerStyle);
            for (int j = 0; j < 5; j++)
                DrawHeatCell(new Rect(left + j * (cellW + gap), top + i * (cellH + gap), cellW, cellH), i * 5 + j, scale);
        }

        // Hidden->Output als zweite Reihe unter der 3x5-Matrix.
        float y = top + 3f * (cellH + gap) + 7f;
        GUI.Label(new Rect(area.x + 15f, y, 105f, 25f), "H -> OUT", smallStyle);
        for (int j = 0; j < 5; j++)
            DrawHeatCell(new Rect(left + j * (cellW + gap), y, cellW, 25f), 15 + j, scale);
        GUI.Label(new Rect(area.x + 14f, area.yMax - 25f, area.width - 28f, 20f),
            "Blau: positiv   |   Rot: negativ   |   Dunkel: nahe Null   |   Gelb: ausgewaehlt", smallStyle);
    }

    private void DrawHeatCell(Rect rect, int index, float scale)
    {
        float value = heatmapShowChanges ? weightChanges[index] : WeightAt(index);
        float intensity = Mathf.Clamp01(Mathf.Abs(value) / scale);
        Color neutral = new Color(0.16f, 0.19f, 0.24f);
        Color target = value >= 0f ? new Color(0.16f, 0.55f, 0.95f) : new Color(0.95f, 0.25f, 0.28f);
        Color old = GUI.color;
        GUI.color = Color.Lerp(neutral, target, intensity);
        GUI.DrawTexture(rect, whiteTexture);
        if (index == selectedWeight)
        {
            GUI.color = Color.yellow;
            GUI.DrawTexture(new Rect(rect.x, rect.y, rect.width, 2f), whiteTexture);
            GUI.DrawTexture(new Rect(rect.x, rect.yMax - 2f, rect.width, 2f), whiteTexture);
            GUI.DrawTexture(new Rect(rect.x, rect.y, 2f, rect.height), whiteTexture);
            GUI.DrawTexture(new Rect(rect.xMax - 2f, rect.y, 2f, rect.height), whiteTexture);
        }
        GUI.color = old;
        string number = heatmapShowChanges ? value.ToString("+0.0000;-0.0000;0.0000") : value.ToString("+0.00;-0.00;0.00");
        GUI.Label(rect, number, smallStyle);
        if (GUI.Button(new Rect(rect.x, rect.y, rect.width, rect.height), GUIContent.none, GUIStyle.none))
            selectedWeight = index;
    }

    private void DrawHistory(Rect area)
    {
        Color old = GUI.color;
        GUI.color = new Color(0.10f, 0.13f, 0.18f, 1f);
        GUI.DrawTexture(area, whiteTexture);
        GUI.color = old;
        List<float> values = histories[selectedWeight];
        if (values.Count < 2) return;
        float min = values[0], max = values[0];
        foreach (float value in values) { min = Mathf.Min(min, value); max = Mathf.Max(max, value); }
        float padding = Mathf.Max(0.01f, (max - min) * 0.12f);
        min -= padding;
        max += padding;
        for (int i = 1; i < values.Count; i++)
        {
            Vector2 a = new Vector2(area.x + (i - 1f) / (values.Count - 1) * area.width,
                area.yMax - (values[i - 1] - min) / (max - min) * area.height);
            Vector2 b = new Vector2(area.x + (float)i / (values.Count - 1) * area.width,
                area.yMax - (values[i] - min) / (max - min) * area.height);
            DrawLine(a, b, new Color(0.3f, 0.95f, 0.65f), 2f);
        }
        GUI.Label(new Rect(area.x + 4, area.y + 2, 120, 18), max.ToString("F3"), smallStyle);
        GUI.Label(new Rect(area.x + 4, area.yMax - 20, 120, 18), min.ToString("F3"), smallStyle);
        GUI.Label(new Rect(area.xMax - 140, area.yMax - 20, 136, 18),
            $"Epoche {historyEpochs[historyEpochs.Count - 1]}", smallStyle);
    }

    private void DrawSensorSlider(string name, int index)
    {
        GUILayout.BeginHorizontal();
        GUILayout.Label(name, GUILayout.Width(55));
        sensorValues[index] = GUILayout.HorizontalSlider(sensorValues[index], 0f, 1f);
        GUILayout.Label(sensorValues[index].ToString("F2"), GUILayout.Width(45));
        GUILayout.EndHorizontal();
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
        Rect rect = new Rect(p.x - nodeSize.x * 0.5f, p.y - nodeSize.y * 0.5f, nodeSize.x, nodeSize.y);
        Color old = GUI.color;
        GUI.color = Color.Lerp(new Color(0.16f, 0.20f, 0.28f),
            new Color(0.1f, 0.75f, 0.4f), Mathf.Clamp01(value));
        GUI.DrawTexture(rect, whiteTexture);
        GUI.color = Color.white;
        GUI.Label(rect, $"{label}\n{value:F2}", smallStyle);
        GUI.color = old;
    }

    private void DrawConnection(Vector2 a, Vector2 b, float weight, int index)
    {
        bool selected = index == selectedWeight;
        Color color = weight >= 0f ? new Color(0.25f, 0.65f, 1f, 0.8f) :
            new Color(1f, 0.35f, 0.35f, 0.8f);
        DrawLine(a, b, selected ? Color.yellow : color,
            selected ? 5f : Mathf.Clamp(Mathf.Abs(weight), 1f, 4f));
        if (showWeights)
        {
            Vector2 mid = (a + b) * 0.5f;
            GUI.Label(new Rect(mid.x - 23, mid.y - 10, 46, 20), weight.ToString("F2"), smallStyle);
        }
    }

    private void DrawLine(Vector2 a, Vector2 b, Color color, float thickness)
    {
        Color old = GUI.color;
        Matrix4x4 previous = GUI.matrix;
        Vector2 delta = b - a;
        float angle = Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg;
        GUI.color = color;
        GUIUtility.RotateAroundPivot(angle, a);
        GUI.DrawTexture(new Rect(a.x, a.y - thickness * 0.5f, delta.magnitude, thickness), whiteTexture);
        GUI.matrix = previous;
        GUI.color = old;
    }
}
