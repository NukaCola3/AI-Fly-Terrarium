using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(Rigidbody))]
public class FlyMovement : MonoBehaviour
{
    [Header("Fluggeschwindigkeit")]
    [SerializeField] private float moveSpeed = 8f;
    [SerializeField] private float rotationSpeed = 120f;
    [SerializeField] private bool manualControl = true;

    private Rigidbody rb;
    private Vector3 moveInput;
    private float turnInput;

    public bool ManualControl => manualControl;
    public Rigidbody Body => rb;

    private void Awake()
    {
        rb = GetComponent<Rigidbody>();
        rb.useGravity = false;
    }

    public void SetManualControl(bool enabled)
    {
        manualControl = enabled;
        SetMovement(Vector3.zero, 0f);
    }

    public void SetMovement(Vector3 localMovement, float turn)
    {
        moveInput = Vector3.ClampMagnitude(localMovement, 1f);
        turnInput = Mathf.Clamp(turn, -1f, 1f);
    }

    public void StopMovement()
    {
        SetMovement(Vector3.zero, 0f);
    }

    private void Update()
    {
        if (!manualControl) return;

        Keyboard keyboard = Keyboard.current;
        if (keyboard == null)
        {
            StopMovement();
            return;
        }

        Vector3 input = Vector3.zero;
        float turn = 0f;
        if (keyboard.wKey.isPressed) input.z += 1f;
        if (keyboard.sKey.isPressed) input.z -= 1f;
        if (keyboard.aKey.isPressed) input.x -= 1f;
        if (keyboard.dKey.isPressed) input.x += 1f;
        if (keyboard.spaceKey.isPressed) input.y += 1f;
        if (keyboard.leftCtrlKey.isPressed) input.y -= 1f;
        if (keyboard.qKey.isPressed) turn -= 1f;
        if (keyboard.eKey.isPressed) turn += 1f;
        SetMovement(input, turn);
    }

    private void FixedUpdate()
    {
        Vector3 movement = transform.right * moveInput.x
                         + Vector3.up * moveInput.y
                         + transform.forward * moveInput.z;

        rb.MovePosition(rb.position + movement * moveSpeed * Time.fixedDeltaTime);
        Quaternion rotation = Quaternion.Euler(0f, turnInput * rotationSpeed * Time.fixedDeltaTime, 0f);
        rb.MoveRotation(rb.rotation * rotation);
    }
}
