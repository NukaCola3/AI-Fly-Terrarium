using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

[RequireComponent(typeof(FlyMovement), typeof(FlySensors), typeof(Rigidbody))]
public class FlyAgent : Agent
{
    [Header("Trainingsbereich")]
    [SerializeField] private Transform spawnPoint;
    [SerializeField] private float collisionPenalty = -1f;

    [Header("Erkundung (Raster im Terrarium)")]
    [Tooltip("Mittelpunkt des Terrariums; leer = Weltursprung (0,0,0).")]
    [SerializeField] private Transform areaCenter;
    [Tooltip("Innere Ausdehnung des Terrariums in Welt-Einheiten (X/Y/Z).")]
    [SerializeField] private Vector3 areaSize = new Vector3(98f, 48f, 48f);
    [SerializeField, Min(1)] private int cellsX = 10;
    [SerializeField, Min(1)] private int cellsY = 5;
    [SerializeField, Min(1)] private int cellsZ = 5;
    [SerializeField] private float newCellReward = 0.05f;
    [SerializeField] private float decisionPenalty = -0.00005f;

    [Header("Diagnose")]
    [SerializeField] private bool logEpisodeSummary = true;

    private FlyMovement movement;
    private FlySensors sensors;
    private Rigidbody body;
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private Vector3 previousPosition;
    private readonly HashSet<Vector3Int> visitedCells = new HashSet<Vector3Int>();
    private float episodeDistance;
    private int episodeCollisions;
    private int episodeDecisions;
    private int episodeNumber;
    private bool episodeStarted;
    private int episodeNewCells;
    private string lastEndReason = "Schrittlimit / extern";

    public override void Initialize()
    {
        movement = GetComponent<FlyMovement>();
        sensors = GetComponent<FlySensors>();
        body = GetComponent<Rigidbody>();
        initialPosition = transform.position;
        initialRotation = transform.rotation;
        movement.SetManualControl(false);
        previousPosition = body.position;
    }

    public override void OnEpisodeBegin()
    {
        if (episodeStarted)
            RecordPreviousEpisode();

        episodeNumber++;
        episodeStarted = true;
        episodeDistance = 0f;
        episodeNewCells = 0;
        episodeCollisions = 0;
        episodeDecisions = 0;
        visitedCells.Clear();
        lastEndReason = "Schrittlimit / extern";

        movement.StopMovement();
        body.linearVelocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;
        body.position = spawnPoint != null ? spawnPoint.position : initialPosition;
        body.rotation = spawnPoint != null ? spawnPoint.rotation : initialRotation;
        previousPosition = body.position;

        // Startzelle zählt als besucht, gibt aber keine Belohnung.
        if (TryGetCell(body.position, out Vector3Int startCell))
            visitedCells.Add(startCell);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        float[] distances = sensors.GetSensorDistances();
        foreach (float distance in distances)
            sensor.AddObservation(distance);

        Vector3 localVelocity = transform.InverseTransformDirection(body.linearVelocity);
        sensor.AddObservation(Vector3.ClampMagnitude(localVelocity / 10f, 1f));
        sensor.AddObservation(transform.forward);

        // v0.4.0: Position im selben weltachsenparallelen Bereich wie das Raster.
        // Mittelpunkt = 0, untere Grenze = -1, obere Grenze = +1.
        sensor.AddObservation(GetNormalizedPosition(body.position));
    }

    private Vector3 GetNormalizedPosition(Vector3 worldPosition)
    {
        Vector3 center = areaCenter != null ? areaCenter.position : Vector3.zero;
        Vector3 offset = worldPosition - center;
        return new Vector3(
            NormalizePositionAxis(offset.x, areaSize.x),
            NormalizePositionAxis(offset.y, areaSize.y),
            NormalizePositionAxis(offset.z, areaSize.z));
    }

    private static float NormalizePositionAxis(float offset, float size)
    {
        // Ungültige Bereichsgrößen dürfen keine Division durch null erzeugen.
        return size > 0f ? Mathf.Clamp(2f * offset / size, -1f, 1f) : 0f;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var a = actions.ContinuousActions;
        movement.SetMovement(new Vector3(a[0], a[1], a[2]), a[3]);

        episodeDecisions++;
        AddReward(decisionPenalty);

        // Die aktuelle Position wird nach jeder Physikaktualisierung erfasst.
        // Hier werden nur Entscheidungen und die geringe Zeitstrafe gezählt.
    }

    private void FixedUpdate()
    {
        if (!episodeStarted) return;

        Vector3 currentPosition = body.position;
        episodeDistance += Vector3.Distance(currentPosition, previousPosition);
        previousPosition = currentPosition;

        if (TryGetCell(currentPosition, out Vector3Int cell) && visitedCells.Add(cell))
        {
            episodeNewCells++;
            AddReward(newCellReward);
        }
    }

    private bool TryGetCell(Vector3 worldPosition, out Vector3Int cell)
    {
        cell = default;
        if (areaSize.x <= 0f || areaSize.y <= 0f || areaSize.z <= 0f)
            return false;

        Vector3 center = areaCenter != null ? areaCenter.position : Vector3.zero;
        Vector3 min = center - areaSize * 0.5f;
        Vector3 normalized = new Vector3(
            (worldPosition.x - min.x) / areaSize.x,
            (worldPosition.y - min.y) / areaSize.y,
            (worldPosition.z - min.z) / areaSize.z);

        if (normalized.x < 0f || normalized.x >= 1f ||
            normalized.y < 0f || normalized.y >= 1f ||
            normalized.z < 0f || normalized.z >= 1f)
            return false;

        cell = new Vector3Int(
            Mathf.Min(Mathf.FloorToInt(normalized.x * cellsX), cellsX - 1),
            Mathf.Min(Mathf.FloorToInt(normalized.y * cellsY), cellsY - 1),
            Mathf.Min(Mathf.FloorToInt(normalized.z * cellsZ), cellsZ - 1));
        return true;
    }

    private void OnCollisionEnter(Collision collision)
    {
        episodeCollisions++;
        lastEndReason = "Kollision mit " + collision.gameObject.name;
        AddReward(collisionPenalty);
        EndEpisode();
    }

    private void RecordPreviousEpisode()
    {
        var recorder = Academy.Instance.StatsRecorder;
        recorder.Add("Fly/VisitedCells", visitedCells.Count);
        recorder.Add("Fly/NewCells", episodeNewCells);
        recorder.Add("Fly/Distance", episodeDistance);
        recorder.Add("Fly/Collisions", episodeCollisions);
        recorder.Add("Fly/Decisions", episodeDecisions);
        recorder.Add("Fly/CollisionRate", episodeCollisions > 0 ? 1f : 0f);

        if (logEpisodeSummary)
        {
            Debug.Log($"AI_Fly Episode {episodeNumber}: " +
                      $"Zellen={visitedCells.Count}, NeueZellen={episodeNewCells}, Strecke={episodeDistance:F1}, " +
                      $"Kollisionen={episodeCollisions}, Entscheidungen={episodeDecisions}, " +
                      $"Ende={lastEndReason}");
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var actions = actionsOut.ContinuousActions;
        actions[0] = 0f;
        actions[1] = 0f;
        actions[2] = 0f;
        actions[3] = 0f;
    }
}
