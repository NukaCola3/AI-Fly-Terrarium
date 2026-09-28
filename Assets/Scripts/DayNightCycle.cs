using UnityEngine;

public class DayNightCycle : MonoBehaviour
{
    [Header("Lights")]
    public Light sun;
    public Light moon;

    [Header("Time")]
    [Range(0f, 24f)]
    public float timeOfDay = 12f;

    public float dayLengthInSeconds = 60f;

    [Header("Sun")]
    public float sunMaxIntensity = 1.2f;

    [Header("Moon")]
    public float moonMaxIntensity = 0.15f;

    void Update()
    {
        // Zeit fortschreiten lassen
        timeOfDay += (24f / dayLengthInSeconds) * Time.deltaTime;

        if (timeOfDay >= 24f)
            timeOfDay -= 24f;

        UpdateLighting();
    }

    void UpdateLighting()
    {
        // Sonne bewegt sich einmal pro Tag um die Welt
        float sunAngle = (timeOfDay / 24f) * 360f - 90f;

        sun.transform.rotation = Quaternion.Euler(sunAngle, 170f, 0f);

        // Sonnenhöhe bestimmen
        float sunHeight = Mathf.Sin((timeOfDay / 24f) * Mathf.PI * 2f);

        // Sonne nur tagsüber aktiv
        float daylight = Mathf.Clamp01(sunHeight);

        sun.intensity = daylight * sunMaxIntensity;

        // Mond gegenüber der Sonne
        moon.transform.rotation =
            Quaternion.Euler(sunAngle + 180f, 170f, 0f);

        float moonLight = Mathf.Clamp01(-sunHeight);

        moon.intensity = moonLight * moonMaxIntensity;
    }
}