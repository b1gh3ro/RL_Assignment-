"""
Project 7 - Self-Driving Lab Simulator
Tabular Q-learning agent that chooses temperature and concentration
to maximise the yield of a simulated chemical reaction.

Baselines: random policy and greedy hill-climbing heuristic.
Run:  python lab_rl.py      (writes figures + results.json into ./out)
"""
import json
import os
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
T_GRID = np.linspace(20, 120, 11)      # temperature, deg C   (10 deg steps)
C_GRID = np.linspace(0.1, 2.1, 11)     # concentration, mol/L (0.2 M steps)
N_T, N_C = len(T_GRID), len(C_GRID)

# actions: 0 T+, 1 T-, 2 C+, 3 C-, 4 repeat current recipe
ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
ACTION_NAMES = ["T+10C", "T-10C", "C+0.2M", "C-0.2M", "repeat"]


def true_yield(T, C):
    """Hidden yield landscape (%), unknown to the agent.
    Global optimum ~ (90 C, 1.5 M); a decoy local optimum ~ (40 C, 0.5 M);
    above 100 C the product decomposes."""
    main = 85 * np.exp(-((T - 90) / 28) ** 2 - ((C - 1.5) / 0.65) ** 2)
    local = 45 * np.exp(-((T - 40) / 12) ** 2 - ((C - 0.5) / 0.3) ** 2)
    decomp = np.where(T > 100, 2.0 * (T - 100), 0.0)
    return np.clip(main + local - decomp + 5, 0, 100)


class LabEnv:
    """One episode = one experimental campaign with a fixed budget of runs.
    State   : (temperature index, concentration index) of the current recipe
    Action  : move one grid step in T or C, or repeat the recipe
    Reward  : measured yield / 100 (noisy) minus an energy/safety cost for hot runs
    """

    def __init__(self, budget=30, noise=3.0, seed=None):
        self.budget = budget
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.t = 0
        self.i = int(self.rng.integers(N_T))
        self.j = int(self.rng.integers(N_C))
        return (self.i, self.j)

    def measure(self, i, j):
        y = true_yield(T_GRID[i], C_GRID[j]) + self.rng.normal(0, self.noise)
        return float(np.clip(y, 0, 100))

    def step(self, a):
        di, dj = ACTIONS[a]
        self.i = int(np.clip(self.i + di, 0, N_T - 1))
        self.j = int(np.clip(self.j + dj, 0, N_C - 1))
        y = self.measure(self.i, self.j)
        energy_cost = 0.05 * max(0.0, (T_GRID[self.i] - 80) / 40)
        r = y / 100.0 - energy_cost
        self.t += 1
        done = self.t >= self.budget
        return (self.i, self.j), r, done, {"yield": y,
                                           "true": true_yield(T_GRID[self.i], C_GRID[self.j])}


# ----------------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------------
class QLearner:
    def __init__(self, alpha=0.5, gamma=0.97, eps_start=1.0, eps_end=0.05,
                 eps_decay_eps=1500, seed=0):
        self.Q = np.zeros((N_T, N_C, len(ACTIONS)))
        self.N = np.zeros_like(self.Q)          # visit counts for step-size decay
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_eps
        self.rng = np.random.default_rng(seed)
        self.episode = 0

    @property
    def eps(self):
        frac = min(1.0, self.episode / self.eps_decay)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(len(ACTIONS)))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, s, a, r, s2, done):
        target = r + (0.0 if done else self.gamma * self.Q[s2].max())
        self.N[s + (a,)] += 1
        lr = max(0.02, self.alpha / np.sqrt(self.N[s + (a,)]))   # decaying step size averages out noise
        self.Q[s + (a,)] += lr * (target - self.Q[s + (a,)])


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, s):
        pass

    def act(self, s, last=None):
        return int(self.rng.integers(len(ACTIONS)))


class HillClimber:
    """Heuristic a chemist might use: try a neighbouring recipe; if the yield
    went down, step back and try another direction; once no direction helps,
    keep repeating the best recipe."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, s):
        self.best_y = -np.inf
        self.best_s = s
        self.untried = list(self.rng.permutation(4))
        self.returning = None

    def act(self, s, last_y=None):
        if last_y is not None:
            if last_y > self.best_y:
                self.best_y, self.best_s = last_y, s
                self.untried = list(self.rng.permutation(4))
        if s != self.best_s:                       # walk back to best recipe
            di = np.sign(self.best_s[0] - s[0]); dj = np.sign(self.best_s[1] - s[1])
            return ACTIONS.index((int(di), 0)) if di != 0 else ACTIONS.index((0, int(dj)))
        if self.untried:
            return int(self.untried.pop())
        return 4                                    # local optimum: exploit


# ----------------------------------------------------------------------------
# Experiment loop
# ----------------------------------------------------------------------------
def run_episode(env, agent, kind, learn=True):
    s = env.reset()
    if kind != "q":
        agent.reset(s)
    total, last_y, yields = 0.0, None, []
    done = False
    while not done:
        if kind == "q":
            a = agent.act(s, greedy=not learn)
        else:
            a = agent.act(s, last_y)
        s2, r, done, info = env.step(a)
        if kind == "q" and learn:
            agent.update(s, a, r, s2, done)
        total += r
        last_y = info["yield"]
        yields.append(info["true"])
        s = s2
    if kind == "q" and learn:
        agent.episode += 1
    # final recipe quality = true yield of last 5 runs (what the lab converged on)
    return total, float(np.mean(yields[-5:]))


def smooth(x, k=25):
    x = np.asarray(x)
    if len(x) < k:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, np.nan), out])


def main(n_episodes=3000, seeds=(0, 1, 2, 3, 4)):
    curves = {k: {"ret": [], "final": []} for k in ("q", "random", "hill")}
    q_agents = []
    for sd in seeds:
        for kind in curves:
            env = LabEnv(seed=1000 + sd)
            agent = {"q": QLearner(seed=sd), "random": RandomAgent(sd),
                     "hill": HillClimber(sd)}[kind]
            rets, finals = [], []
            for _ in range(n_episodes):
                R, f = run_episode(env, agent, kind)
                rets.append(R); finals.append(f)
            curves[kind]["ret"].append(rets)
            curves[kind]["final"].append(finals)
            if kind == "q":
                q_agents.append(agent)

    # greedy evaluation of trained Q policies on fresh episodes
    eval_res = {}
    for kind in ("q", "random", "hill"):
        R_all, F_all = [], []
        for sd in seeds:
            env = LabEnv(seed=5000 + sd)
            agent = q_agents[sd] if kind == "q" else (RandomAgent(sd) if kind == "random" else HillClimber(sd))
            for _ in range(200):
                R, f = run_episode(env, agent, kind, learn=False)
                R_all.append(R); F_all.append(f)
        eval_res[kind] = {"return_mean": float(np.mean(R_all)), "return_std": float(np.std(R_all)),
                          "final_yield_mean": float(np.mean(F_all)), "final_yield_std": float(np.std(F_all)),
                          "pct_reaching_global_opt": float(np.mean(np.array(F_all) >= 75) * 100)}

    labels = {"q": "Q-learning", "random": "Random baseline", "hill": "Hill-climbing heuristic"}
    colors = {"q": "#1f77b4", "random": "#7f7f7f", "hill": "#d62728"}

    # --- Figure 1: learning curves (return) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax, ylab in (("ret", axes[0], "Episode return (sum of rewards)"),
                             ("final", axes[1], "True yield of final 5 runs (%)")):
        for kind in curves:
            arr = np.array([smooth(r) for r in curves[kind][metric]])
            m, s = np.nanmean(arr, 0), np.nanstd(arr, 0)
            x = np.arange(len(m))
            ax.plot(x, m, color=colors[kind], label=labels[kind], lw=1.8)
            ax.fill_between(x, m - s, m + s, color=colors[kind], alpha=0.15)
        ax.set_xlabel("Episode (experimental campaign)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    axes[0].set_title("Learning curve: return")
    axes[1].set_title("Learning curve: converged yield")
    fig.suptitle("Self-driving lab: Q-learning vs baselines (mean ± std over 5 seeds, 25-ep moving avg)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_curves.png"), dpi=160)
    plt.close(fig)

    # --- Figure 2: yield landscape + greedy trajectories ---
    TT, CC = np.meshgrid(T_GRID, C_GRID, indexing="ij")
    Z = true_yield(TT, CC)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(Z.T, origin="lower", aspect="auto", cmap="viridis",
                   extent=[T_GRID[0] - 5, T_GRID[-1] + 5, C_GRID[0] - 0.1, C_GRID[-1] + 0.1])
    plt.colorbar(im, ax=ax, label="True yield (%)")
    agent = q_agents[0]
    for start, col in (((1, 1), "white"), ((0, 9), "orange"), ((10, 0), "red"), ((3, 3), "cyan")):
        i, j = start
        path = [(i, j)]
        for _ in range(30):
            a = int(np.argmax(agent.Q[i, j]))
            di, dj = ACTIONS[a]
            i = int(np.clip(i + di, 0, N_T - 1)); j = int(np.clip(j + dj, 0, N_C - 1))
            path.append((i, j))
        xs = [T_GRID[p[0]] for p in path]; ys = [C_GRID[p[1]] for p in path]
        ax.plot(xs, ys, "-o", color=col, ms=3, lw=1.5)
        ax.plot(xs[0], ys[0], "s", color=col, ms=8)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Concentration (mol/L)")
    ax.set_title("Hidden yield landscape and learned greedy paths\n(squares = start recipe)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "policy_paths.png"), dpi=160)
    plt.close(fig)

    # --- Figure 3: bar chart of evaluation ---
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ks = ["random", "hill", "q"]
    ax.bar([labels[k] for k in ks], [eval_res[k]["final_yield_mean"] for k in ks],
           yerr=[eval_res[k]["final_yield_std"] for k in ks], color=[colors[k] for k in ks], capsize=5)
    ax.set_ylabel("Converged true yield (%)")
    ax.set_title("Evaluation on 1000 fresh campaigns (greedy policy)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "evaluation_bar.png"), dpi=160)
    plt.close(fig)

    best = np.unravel_index(np.argmax(Z), Z.shape)
    results = {"evaluation": eval_res,
               "global_optimum": {"T": float(T_GRID[best[0]]), "C": float(C_GRID[best[1]]),
                                  "yield": float(Z[best])},
               "episodes": n_episodes, "seeds": list(seeds)}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
