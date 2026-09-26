"""
Run every Biped test (each in its own process), in each robot configuration that matters.
    python tests/run_all.py
Exit code 0 only if everything passed. Rebuild bereshitCore first after changing the C++ engine.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [
    ("test_engine_physics.py", {}),
    ("test_engine_contacts.py", {}),
    ("test_engine_joints.py", {}),
    ("test_parallel_worker.py", {}),
    ("test_fall_report.py", {}),
    ("test_geometry_config.py", {}),
    ("test_geometry_config.py", {"BIPED_BUILD": "current"}),
    ("test_geometry_config.py", {"BIPED_SERVO": "STS3095_12V"}),
    ("test_robot.py", {}),
    ("test_robot.py", {"BIPED_HIP_YAW": "0"}),
    ("test_ppo.py", {}),
    ("test_ppo_edges.py", {}),
    ("test_curriculum.py", {}),
    ("test_reward.py", {}),
    ("test_walk_stats.py", {}),
    ("test_reference_motions.py", {}),
    ("test_symmetry.py", {}),
    ("test_motors.py", {}),
    ("test_one_leg_strength.py", {}),
    ("test_one_leg_strength.py", {"BIPED_BUILD": "v2"}),
    ("test_one_leg_strength.py", {"BIPED_BUILD": "current"}),
    ("test_one_leg_strength.py", {"BIPED_BUILD": "hybrid"}),
    ("test_geometry_config.py", {"BIPED_BUILD": "hybrid"}),
    ("test_symmetry.py", {"BIPED_HIP_YAW": "0"}),
]

results = []
for script, env in RUNS:
    label = script + (" [" + ", ".join(f"{k}={v}" for k, v in env.items()) + "]" if env else "")
    print(f"\n=== {label}", flush=True)
    proc = subprocess.run([sys.executable, "-u", os.path.join(HERE, script)], env={**os.environ, **env})
    results.append((label, proc.returncode == 0))

print("\n=== summary")
for label, ok in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
sys.exit(0 if all(ok for _, ok in results) else 1)
