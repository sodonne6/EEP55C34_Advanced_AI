# Auto-imported by Python at startup if on PYTHONPATH.
# Patch MediaPipe so pose-format can do: mp.solutions.holistic

try:
    import mediapipe as mp
    try:
        from mediapipe.python import solutions as mp_solutions
    except Exception:
        import mediapipe.solutions as mp_solutions

    mp.solutions = mp_solutions
except Exception:
    # If mediapipe isn't installed, don't crash interpreter startup
    pass
