
using UnityEngine;

public class FlySensors : MonoBehaviour
{
    [Header("Sensor-Einstellungen")]
    [SerializeField] private float sensorRange = 15f;
    [SerializeField] private LayerMask obstacleLayers = ~0;
    [SerializeField] private bool showSensorRays = true;

    private readonly float[] distances = new float[10];

    public float[] GetSensorDistances()
    {
        return (float[])distances.Clone();
    }

    private void FixedUpdate()
    {
        ScanEnvironment();
    }

    private void ScanEnvironment()
    {
        Vector3[] directions =
        {
            transform.forward,
            -transform.forward,
            transform.right,
            -transform.right,

            (transform.forward + transform.right).normalized,
            (transform.forward - transform.right).normalized,
            (-transform.forward + transform.right).normalized,
            (-transform.forward - transform.right).normalized,

            Vector3.up,
            Vector3.down
        };

        Vector3 origin = transform.position;

        for (int i = 0; i < directions.Length; i++)
        {
            Vector3 direction = directions[i];

            bool hitSomething = Physics.Raycast(
                origin,
                direction,
                out RaycastHit hit,
                sensorRange,
                obstacleLayers,
                QueryTriggerInteraction.Ignore
            );

            float distance = hitSomething
                ? hit.distance
                : sensorRange;

            // Normalisierter Wert zwischen 0 und 1
            distances[i] = distance / sensorRange;

            if (showSensorRays)
            {
                Debug.DrawRay(
                    origin,
                    direction * distance,
                    hitSomething ? Color.red : Color.green
                );
            }
        }
    }
}
