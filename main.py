import os
import numpy as np


a = 0.0
b = 1.0


def F_uniform(x: float, k_potential_walls: float) -> float:
    if x < a:
        return -k_potential_walls * (x - a)
    elif x > b:
        return -k_potential_walls * (x - b)
    else:
        return 0.0


def sample_langevin(n_steps: int, eps: float, k: float, x0: float = 0.5) -> float:
    x = x0
    for _ in range(n_steps):
        z = np.random.normal(0, 1)
        x += F_uniform(x, k) + np.sqrt(2 * eps) * z
    return x


def trace_langevin(n_steps: int, eps: float, k: float, x0: float = 0.5) -> np.ndarray:
    x = x0
    traj = np.empty(n_steps + 1)
    traj[0] = x
    for t in range(1, n_steps + 1):
        z = np.random.normal(0, 1)
        x += F_uniform(x, k) + np.sqrt(2 * eps) * z
        traj[t] = x
    return traj


if __name__ == "__main__":

    n_steps = 30_000
    eps = 1e-5
    k = 3.0
    n_samples = 10_000

    np.random.seed(42)

    samples = [sample_langevin(n_steps=n_steps, eps=eps, k=k) for _ in range(n_samples)]
    samples = np.asarray(samples)

    trace_len = 5_000
    trace = trace_langevin(n_steps=trace_len, eps=eps, k=k)

    os.makedirs("plots", exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(
            samples,
            bins=50,
            range=(a, b),
            density=True,
            alpha=0.7,
            color="#4C78A8",
            edgecolor="white",
        )
        ax.hlines(
            y=1.0 / (b - a),
            xmin=a,
            xmax=b,
            colors="#F58518",
            linestyles="--",
            label="Uniform density",
        )
        ax.set_title(
            f"Langevin samples in [{a}, {b}]\nsteps={n_steps}, eps={eps}, k={k}, N={n_samples}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.legend()
        fig.tight_layout()
        fig.savefig("plots/samples_hist.png", dpi=150)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.plot(trace, lw=1.0, color="#54A24B")
        ax2.set_title(f"Single-chain trace (n_steps={trace_len})")
        ax2.set_xlabel("step")
        ax2.set_ylabel("x")
        ax2.set_ylim(a - 0.1, b + 0.1)
        ax2.axhline(a, color="gray", lw=0.8, ls=":")
        ax2.axhline(b, color="gray", lw=0.8, ls=":")
        fig2.tight_layout()
        fig2.savefig("plots/trace.png", dpi=150)
        plt.close(fig2)

    except Exception as e:
        print(
            {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "min": float(np.min(samples)),
                "max": float(np.max(samples)),
            }
        )
