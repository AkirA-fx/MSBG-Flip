"""
examples/test_flip.py

Phase 5c: Python から MSBG-Flip を駆動するエンドツーエンドの smoke test。
"""
import os
import sys
import time

import numpy as np


def make_sphere_sdf(sx, sy, sz, cx, cy, cz, radius):
    """Return ndarray (sx, sy, sz) float32 SDF of a sphere."""
    xs, ys, zs = np.meshgrid(
        np.arange(sx), np.arange(sy), np.arange(sz), indexing="ij"
    )
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2) - radius
    return dist.astype(np.float32)


def main():
    import msbg_flip

    print(f"msbg_flip module: {msbg_flip.__file__}")

    sim = msbg_flip.FlipSimulation(resolution=64, block_size=16)
    print(
        f"sim created: {sim.sx}x{sim.sy}x{sim.sz}, block={sim.block_size}"
    )

    sim.configure(
        rho_l=1000.0,
        rho_g=1.0,
        cfl=3.0,
        flip_alpha=0.95,
        solver="MIC0_PCG",
        use_float16=True,
        enable_debug_slice=False,  # disable PNG slice output during test
    )

    # Sphere collider at (32, 24, 32) radius 10 — same as Phase 5b sanity test
    sdf = make_sphere_sdf(sim.sx, sim.sy, sim.sz, 32, 24, 32, 10.0)
    sim.set_collider(sdf, eps=0.5, push_out=1.0)

    n = 5  # short run for smoke test
    t0 = time.time()
    rc = sim.run(n)
    elapsed = time.time() - t0
    print(f"sim.run({n}) -> rc={rc}, elapsed={elapsed:.2f}s")

    # Result inspection
    density = sim.get_density()
    velocity = sim.get_velocity()
    particles = sim.get_particles()

    assert density.shape == (sim.sx, sim.sy, sim.sz)
    assert density.dtype == np.float32
    assert velocity.shape == (sim.sx, sim.sy, sim.sz, 3)
    assert velocity.dtype == np.float32
    assert "pos" in particles and "vel" in particles and "phase" in particles

    print(
        f"density: shape={density.shape} dtype={density.dtype} "
        f"min={density.min():.3f} max={density.max():.3f} "
        f"nonzero={(density > 0).sum()}"
    )
    print(
        f"velocity: shape={velocity.shape} "
        f"vmag_max={np.linalg.norm(velocity, axis=-1).max():.3f}"
    )
    n_parts = particles["pos"].shape[0]
    n_liquid = (particles["phase"] == 0).sum()
    n_air = (particles["phase"] == 1).sum()
    print(
        f"particles: total={n_parts} liquid={n_liquid} air={n_air}"
    )

    # Sanity: collider region should have no particles inside (after run)
    px = particles["pos"]
    inside_mask = np.sqrt(
        (px[:, 0] - 32) ** 2 + (px[:, 1] - 24) ** 2 + (px[:, 2] - 32) ** 2
    ) < 10.0
    n_inside = int(inside_mask.sum())
    print(f"particles inside sphere collider: {n_inside}")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
