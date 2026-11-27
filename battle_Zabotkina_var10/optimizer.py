"""
Гибридный метод оптимизации: ACO на сетке + последовательное сгущение сетки, вариант № 10.
Автор: Заботкина М. А., группа K3339 (поток МетОпт 1.2).

Обязательный интерфейс (не менять):
    optimize(f, bounds, max_nfe=10_000, seed=None, constraints=None, **params) -> dict

Возвращаемый словарь:
    x         — np.ndarray shape (2,) — найденная точка (обязательно)
    f         — float — значение f(x) (обязательно)
    nfe       — int — число вычислений f (обязательно)
    history   — list[float] — лучшее значение после каждого вычисления f (для графика сходимости)
    switches  — list[(int, str)] — моменты переключения методов: (nfe, "название этапа")
    snapshots — list[(int, str, np.ndarray)] — положения популяции/роя: (nfe, подпись, точки (n, 2))

Правила:
  * все компоненты гибрида — собственная реализация; scipy.optimize и другие готовые оптимизаторы запрещены;
  * не импортируйте functions.py и не используйте известный минимум: метод знает только f и bounds;
  * никакой печати и рисования внутри optimize (это искажает время);
  * все случайные числа — только из rng = np.random.default_rng(seed);
  * не превышать max_nfe: на битве лишнее вычисление прерывает запуск (запуск — неуспешный);
  * параметры метода задаются значениями по умолчанию ниже и на защите не меняются.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------- параметры метода
DEFAULTS = dict(
    grid_size=23,
    n_ants=25,
    evap=0.25,
    alpha=1.2,
    tau_min=0.05,
    tau_max=10.0,
    zoom_factor=0.50,
    max_levels=25,
    level0_min_iters=60,
    level0_stagnation=15,
    min_iters=5,
    stagnation_iters=5,
    max_iters_per_level=30,
    eps_explore=0.20,
    use_aco=True,
)


class Budget:
    """Учёт вычислений, лучшей точки и истории. Используйте self.eval вместо f."""

    def __init__(self, f, bounds, max_nfe, constraints=None, penalty=1e6):
        self.f, self.max_nfe = f, max_nfe
        self.lo = np.array([b[0] for b in bounds], dtype=float)
        self.hi = np.array([b[1] for b in bounds], dtype=float)
        self.constraints = list(constraints or [])
        self.penalty = penalty
        self.nfe = 0
        self.best_x, self.best_f = None, np.inf
        self.history, self.switches, self.snapshots = [], [], []

    @property
    def left(self) -> int:
        return self.max_nfe - self.nfe

    def clip(self, x):
        return np.clip(x, self.lo, self.hi)

    def violation(self, x):
        return sum(max(0.0, g(x)) for g in self.constraints)

    def eval(self, x) -> float:
        """Значение со штрафом за нарушение ограничений; inf, если бюджет исчерпан."""
        if self.nfe >= self.max_nfe:
            return np.inf
        x = self.clip(np.asarray(x, dtype=float))
        self.nfe += 1
        fx = float(self.f(x))
        v = self.violation(x)
        val = fx + self.penalty * v
        if v == 0 and fx < self.best_f:
            self.best_f, self.best_x = fx, x.copy()
        self.history.append(self.best_f)
        return val

    def stage(self, name):
        self.switches.append((self.nfe, name))

    def snap(self, name, points):
        self.snapshots.append((self.nfe, name, np.array(points, dtype=float).copy()))

    def result(self):
        x = self.best_x if self.best_x is not None else (self.lo + self.hi) / 2
        return {"x": x, "f": self.best_f, "nfe": self.nfe, "history": self.history,
                "switches": self.switches, "snapshots": self.snapshots}


# ------------------------------------------------------------- компоненты гибрида
def global_stage(B: Budget, rng, **p):
    """Глобальный этап: ACO на начальной сетке по всей области D."""
    B.stage("глобальный поиск")
    m = p["grid_size"]
    n_ants = p["n_ants"]
    evap = p["evap"]
    alpha = p["alpha"]
    tau_min = p["tau_min"]
    tau_max = p["tau_max"]
    eps_explore = p["eps_explore"]
    use_aco = p["use_aco"]

    gx1 = np.linspace(B.lo[0], B.hi[0], m)
    gx2 = np.linspace(B.lo[1], B.hi[1], m)
    step = (B.hi - B.lo) / (m - 1)
    tau1 = np.ones(m)
    tau2 = np.ones((m, m))
    visited = set()

    best_lvl_val = np.inf
    best_lvl_ij = None
    no_improve = 0
    is_only_global = (p["max_levels"] == 1)
    last_pts = None

    it = 0
    max_it = 400 if is_only_global else 80
    while B.left > 0 and it < max_it:
        p1 = tau1 ** alpha
        p1 = p1 / p1.sum()

        paths = []
        pts = []
        vals = []

        for k in range(n_ants):
            if B.left <= 0:
                break
            if rng.random() < eps_explore:
                i = int(rng.integers(0, m))
                j = int(rng.integers(0, m))
            else:
                i = int(rng.choice(m, p=p1))
                p2 = tau2[i] ** alpha
                p2 = p2 / p2.sum()
                j = int(rng.choice(m, p=p2))

            if (i, j) not in visited:
                visited.add((i, j))
                x = np.array([gx1[i], gx2[j]], dtype=float)
            else:
                x = np.array([gx1[i], gx2[j]], dtype=float) + rng.uniform(-0.5 * step, 0.5 * step)

            x = B.clip(x)
            if B.constraints and B.best_x is not None and B.violation(x) > 0:
                for _ in range(5):
                    x = 0.5 * (x + B.best_x)
                    if B.violation(x) == 0:
                        break

            val = B.eval(x)

            if val < best_lvl_val:
                best_lvl_val = val
                best_lvl_ij = (i, j)
                no_improve = -1
                for d in (0, 1):
                    if B.left <= 0:
                        break
                    for scale in (0.2, 0.05, 0.01, 0.002, 0.0004):
                        if B.left <= 0:
                            break
                        for sgn in (-1.0, 1.0):
                            if B.left <= 0:
                                break
                            dx = np.zeros(2)
                            dx[d] = sgn * scale * step[d]
                            x_probe = B.clip(x + dx)
                            if B.constraints and B.best_x is not None and B.violation(x_probe) > 0:
                                for _ in range(5):
                                    x_probe = 0.5 * (x_probe + B.best_x)
                                    if B.violation(x_probe) == 0:
                                        break
                            val_probe = B.eval(x_probe)
                            if val_probe < best_lvl_val:
                                x, val = x_probe, val_probe
                                best_lvl_val = val
                                break

            paths.append((i, j))
            pts.append(x)
            vals.append(val)

        if len(pts) == 0:
            break

        last_pts = np.array(pts)
        if it == 0 and len(B.snapshots) == 0:
            B.snap("старт", last_pts)

        if use_aco:
            tau1 = tau1 * (1.0 - evap)
            tau2 = tau2 * (1.0 - evap)

            order = np.argsort(vals)
            k_elite = max(3, len(vals) // 3)
            for rank in range(min(k_elite, len(order))):
                idx = order[rank]
                if not np.isfinite(vals[idx]):
                    continue
                w = (k_elite - rank) / k_elite
                i, j = paths[idx]
                tau1[i] += w
                tau2[i, j] += w
                tau2[:, j] += 0.3 * w

            if best_lvl_ij is not None:
                bi, bj = best_lvl_ij
                tau1[bi] += 0.5
                tau2[bi, bj] += 0.5

            tau1 = np.clip(tau1, tau_min, tau_max)
            tau2 = np.clip(tau2, tau_min, tau_max)

        no_improve += 1
        it += 1
        if not is_only_global and it >= p["level0_min_iters"] and no_improve >= p["level0_stagnation"]:
            break

    if last_pts is not None and len(B.snapshots) <= 1:
        B.snap("конец глобального этапа", last_pts)


def local_stage(B: Budget, rng, **p):
    """Этап уточнения: последовательное сгущение сетки вокруг лучшего узла."""
    if p["max_levels"] <= 1 or B.left <= 0:
        return

    B.stage("локальное уточнение")
    m = p["grid_size"]
    n_ants = p["n_ants"]
    evap = p["evap"]
    alpha = p["alpha"]
    tau_min = p["tau_min"]
    tau_max = p["tau_max"]
    zoom_factor = p["zoom_factor"]
    max_levels = p["max_levels"]
    eps_explore = p["eps_explore"]
    use_aco = p["use_aco"]

    last_pts = None

    while B.left > 0:
        cur_lo = B.lo.copy()
        cur_hi = B.hi.copy()
        best_any_x = None
        best_any_val = np.inf

        for level in range(1, max_levels):
            if B.left <= 0 or np.max(cur_hi - cur_lo) < 1e-8:
                break

            center = B.best_x if B.best_x is not None else best_any_x
            if center is None:
                center = 0.5 * (cur_lo + cur_hi)

            half = 0.5 * (cur_hi - cur_lo) * zoom_factor
            new_lo = np.maximum(B.lo, center - half)
            new_hi = np.minimum(B.hi, center + half)
            for d in range(2):
                span = min(B.hi[d] - B.lo[d], 2.0 * half[d])
                if new_lo[d] <= B.lo[d]:
                    new_lo[d] = B.lo[d]
                    new_hi[d] = B.lo[d] + span
                elif new_hi[d] >= B.hi[d]:
                    new_hi[d] = B.hi[d]
                    new_lo[d] = B.hi[d] - span
            cur_lo, cur_hi = new_lo, new_hi

            gx1 = np.linspace(cur_lo[0], cur_hi[0], m)
            gx2 = np.linspace(cur_lo[1], cur_hi[1], m)
            step = (cur_hi - cur_lo) / (m - 1)
            tau1 = np.ones(m)
            tau2 = np.ones((m, m))
            visited = set()

            best_lvl_val = np.inf
            best_lvl_ij = None
            no_improve = 0
            is_last = (level == max_levels - 1)

            it = 0
            while B.left > 0 and (is_last or it < p["max_iters_per_level"]):
                p1 = tau1 ** alpha
                p1 = p1 / p1.sum()

                paths = []
                pts = []
                vals = []

                for k in range(n_ants):
                    if B.left <= 0:
                        break
                    if rng.random() < eps_explore:
                        i = int(rng.integers(0, m))
                        j = int(rng.integers(0, m))
                    else:
                        i = int(rng.choice(m, p=p1))
                        p2 = tau2[i] ** alpha
                        p2 = p2 / p2.sum()
                        j = int(rng.choice(m, p=p2))

                    if (i, j) not in visited:
                        visited.add((i, j))
                        x = np.array([gx1[i], gx2[j]], dtype=float)
                    else:
                        x = np.array([gx1[i], gx2[j]], dtype=float) + rng.uniform(-0.5 * step, 0.5 * step)

                    x = B.clip(x)
                    if B.constraints and B.best_x is not None and B.violation(x) > 0:
                        for _ in range(5):
                            x = 0.5 * (x + B.best_x)
                            if B.violation(x) == 0:
                                break

                    val = B.eval(x)

                    if val < best_any_val:
                        best_any_val = val
                        best_any_x = x.copy()
                    if val < best_lvl_val:
                        best_lvl_val = val
                        best_lvl_ij = (i, j)
                        no_improve = -1
                        for d in (0, 1):
                            if B.left <= 0:
                                break
                            for scale in (0.2, 0.05, 0.01, 0.002, 0.0004):
                                if B.left <= 0:
                                    break
                                for sgn in (-1.0, 1.0):
                                    if B.left <= 0:
                                        break
                                    dx = np.zeros(2)
                                    dx[d] = sgn * scale * step[d]
                                    x_probe = B.clip(x + dx)
                                    if B.constraints and B.best_x is not None and B.violation(x_probe) > 0:
                                        for _ in range(5):
                                            x_probe = 0.5 * (x_probe + B.best_x)
                                            if B.violation(x_probe) == 0:
                                                break
                                    val_probe = B.eval(x_probe)
                                    if val_probe < best_any_val:
                                        best_any_val = val_probe
                                        best_any_x = x_probe.copy()
                                        x, val = x_probe, val_probe
                                        break

                    paths.append((i, j))
                    pts.append(x)
                    vals.append(val)

                if len(pts) == 0:
                    break

                last_pts = np.array(pts)
                if it == 0 and level == 3 and len(B.snapshots) <= 2:
                    B.snap("середина уточнения", last_pts)

                if use_aco:
                    tau1 = tau1 * (1.0 - evap)
                    tau2 = tau2 * (1.0 - evap)

                    order = np.argsort(vals)
                    k_elite = max(3, len(vals) // 3)
                    for rank in range(min(k_elite, len(order))):
                        idx = order[rank]
                        if not np.isfinite(vals[idx]):
                            continue
                        w = (k_elite - rank) / k_elite
                        i, j = paths[idx]
                        tau1[i] += w
                        tau2[i, j] += w
                        tau2[:, j] += 0.3 * w

                    if best_lvl_ij is not None:
                        bi, bj = best_lvl_ij
                        tau1[bi] += 0.5
                        tau2[bi, bj] += 0.5

                    tau1 = np.clip(tau1, tau_min, tau_max)
                    tau2 = np.clip(tau2, tau_min, tau_max)

                no_improve += 1
                it += 1
                if not is_last and it >= p["min_iters"] and no_improve >= p["stagnation_iters"]:
                    break

        if B.left > 1200:
            global_stage(B, rng, **p)
        else:
            break

    if last_pts is not None:
        B.snap("конец уточнения", last_pts)


# ------------------------------------------------------------- точка входа
def optimize(f, bounds, max_nfe=10_000, seed=None, constraints=None, **params):
    p = {**DEFAULTS, **params}
    rng = np.random.default_rng(seed)
    B = Budget(f, bounds, max_nfe, constraints)
    global_stage(B, rng, **p)
    local_stage(B, rng, **p)
    return B.result()