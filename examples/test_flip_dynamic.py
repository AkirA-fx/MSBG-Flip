"""
examples/test_flip_dynamic.py

Phase 5c-ext smoke test:
  - step-by-step driving (initialize + step loop)
  - dynamic collider (sphere center moves each step)
  - external force field (constant +x wind)
  - add_particles (drop a small batch every 5 steps)
"""
import sys
import time

import numpy as np


def make_sphere_sdf(sx, sy, sz, cx, cy, cz, radius):
    xs, ys, zs = np.meshgrid(
        np.arange(sx), np.arange(sy), np.arange(sz), indexing="ij"
    )
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2) - radius
    return dist.astype(np.float32)


def main():
    import msbg_flip

    sim = msbg_flip.FlipSimulation(resolution=64, block_size=16)
    sim.configure(
        rho_l=1000.0,
        rho_g=1.0,
        cfl=3.0,
        flip_alpha=0.95,
        solver="MIC0_PCG",
        use_float16=True,
        enable_debug_slice=False,
    )
    sim.initialize()
    print(f"after initialize: step_number={sim.step_number}, sx={sim.sx}")

    # Constant +x acceleration ("wind"). Persists across steps until cleared.
    field = np.zeros((sim.sx, sim.sy, sim.sz, 3), dtype=np.float32)
    field[:, :, :, 0] = 4.0  # 4 voxel/sec^2 along +x
    sim.set_external_force(field)

    n_steps = 10
    initial_n = sim.get_particles()["pos"].shape[0]
    t0 = time.time()
    for i in range(n_steps):
        # 動く球: 中心 x が +0.6/step で動く
        cx = 32.0 + 0.6 * i
        sdf = make_sphere_sdf(sim.sx, sim.sy, sim.sz, cx, 24.0, 32.0, 10.0)
        sim.set_collider(sdf, eps=0.5, push_out=1.0)

        # 5 ステップごとに小さな粒子バッチを top から落とす
        if i % 5 == 0:
            n_new = 1024
            pos = np.zeros((n_new, 3), dtype=np.float32)
            pos[:, 0] = np.random.uniform(8, 24, n_new)
            pos[:, 1] = 60.0
            pos[:, 2] = np.random.uniform(24, 40, n_new)
            vel = np.zeros_like(pos)
            phase = np.zeros((n_new,), dtype=np.int32)
            sim.add_particles(pos, vel, phase)

        sim.step()
    elapsed = time.time() - t0
    print(f"loop {n_steps} steps in {elapsed:.2f}s, step_number={sim.step_number}")

    # 検証
    parts = sim.get_particles()
    final_n = parts["pos"].shape[0]
    print(f"particles: initial={initial_n} final={final_n} delta={final_n - initial_n}")
    assert final_n > initial_n, "add_particles should have increased the count"

    velocity = sim.get_velocity()
    vx_mean = velocity[..., 0].mean()
    print(f"vel mean (x): {vx_mean:.3f}  (expected positive due to +x wind)")

    # 風 ON で粒子の x 速度平均が大きい
    pvel = parts["vel"]
    pvx_mean = pvel[:, 0].mean()
    print(f"particle vx mean: {pvx_mean:.3f}")

    # 外力をクリアして 5 ステップ走らせる -> 値が頭打ちにならない
    sim.clear_external_force()
    for _ in range(5):
        sim.step()
    print(f"after clear+5 more steps: step_number={sim.step_number}")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
