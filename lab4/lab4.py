import math
import time
from itertools import product
import numpy as np

assets = ['ЦБ1', 'ЦБ2', 'Депозит']
start_s = np.array([100.0, 800.0, 400.0])
start_cash = 600.0
min_s = np.array([30.0, 150.0, 100.0])

packs = np.array([25.0, 200.0, 100.0])
comm = np.array([0.04, 0.07, 0.05])

market = {
    1: [
        {'name': 'Б', 'p': 0.60, 'r': [1.20, 1.10, 1.07]},
        {'name': 'Н', 'p': 0.30, 'r': [1.05, 1.02, 1.03]},
        {'name': 'П', 'p': 0.10, 'r': [0.80, 0.95, 1.00]}
    ],
    2: [
        {'name': 'Б', 'p': 0.30, 'r': [1.40, 1.15, 1.01]},
        {'name': 'Н', 'p': 0.20, 'r': [1.05, 1.00, 1.00]},
        {'name': 'П', 'p': 0.50, 'r': [0.60, 0.90, 1.00]}
    ],
    3: [
        {'name': 'Б', 'p': 0.40, 'r': [1.15, 1.12, 1.05]},
        {'name': 'Н', 'p': 0.40, 'r': [1.05, 1.01, 1.01]},
        {'name': 'П', 'p': 0.20, 'r': [0.70, 0.94, 1.00]}
    ]
}


def validate_market(mkt):
    for stage in (1, 2, 3):
        if stage not in mkt or len(mkt[stage]) == 0:
            raise ValueError(f"Отсутствуют сценарии для этапа {stage}")
        prob_sum = sum(sc['p'] for sc in mkt[stage])
        if abs(prob_sum - 1.0) > 1e-6:
            raise ValueError(f"Сумма вероятностей на этапе {stage} равна {prob_sum}, ожидалось 1.0")
        for sc in mkt[stage]:
            if len(sc['r']) != 3 or sc['p'] < 0:
                raise ValueError(f"Некорректные данные сценария на этапе {stage}")


def get_actions(s, cash, cur_min_s=min_s):
    if cash < -1e-9:
        return np.empty((0, 3), dtype=int), np.empty(0)

    min_u = [math.ceil((cur_min_s[i] - s[i] - 1e-9) / packs[i]) for i in range(3)]
    max_cash = cash + sum(-min_u[i] * packs[i] * (1 - comm[i]) for i in range(3) if min_u[i] < 0)
    max_u = [int(max_cash // (packs[i] * (1 + comm[i]))) for i in range(3)]

    for i in range(3):
        if min_u[i] > max_u[i]:
            return np.empty((0, 3), dtype=int), np.empty(0)

    u1 = np.arange(min_u[0], max_u[0] + 1)
    u2 = np.arange(min_u[1], max_u[1] + 1)
    u3 = np.arange(min_u[2], max_u[2] + 1)

    g1, g2, g3 = np.meshgrid(u1, u2, u3, indexing='ij')
    acts = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1)

    vals = acts * packs
    comms = np.abs(vals) * comm
    new_cash = cash - np.sum(vals + comms, axis=1)
    new_s = s + vals

    mask = (new_cash >= -1e-9) & np.all(new_s >= cur_min_s - 1e-9, axis=1)
    return acts[mask], new_cash[mask]


def solve_stage3_batch(s_mat, cash_arr):
    r_avg = np.sum([sc['p'] * np.array(sc['r']) for sc in market[3]], axis=0)
    buy_profit = r_avg - (1 + comm)
    sell_profit = (1 - comm) - r_avg

    if np.all(buy_profit <= 0) and np.all(sell_profit <= 0):
        req_u = np.maximum(0, np.ceil((min_s - s_mat - 1e-9) / packs))
        cost = np.sum(req_u * packs * (1 + comm), axis=1)
        new_cash = cash_arr - cost
        new_s = s_mat + req_u * packs
        vals = np.sum(new_s * r_avg, axis=1) + new_cash

        bad_idx = np.where(new_cash < -1e-9)[0]
        for idx in bad_idx:
            acts, nc = get_actions(s_mat[idx], cash_arr[idx])
            if len(acts) > 0:
                ns = s_mat[idx] + acts * packs
                vals[idx] = np.max(np.sum(ns * r_avg, axis=1) + nc)
            else:
                vals[idx] = -np.inf
        return vals
    else:
        res = np.zeros(len(cash_arr))
        for idx in range(len(cash_arr)):
            acts, nc = get_actions(s_mat[idx], cash_arr[idx])
            if len(acts) == 0:
                res[idx] = -np.inf
            else:
                ns = s_mat[idx] + acts * packs
                res[idx] = np.max(np.sum(ns * r_avg, axis=1) + nc)
        return res


def solve_stage3_single(s, cash):
    acts, nc = get_actions(s, cash)
    if len(acts) == 0:
        return -np.inf, None
    r_avg = np.sum([sc['p'] * np.array(sc['r']) for sc in market[3]], axis=0)
    ns = s + acts * packs
    vals = np.sum(ns * r_avg, axis=1) + nc
    best_i = int(np.argmax(vals))
    return float(vals[best_i]), tuple(int(x) for x in acts[best_i])


def solve_stage2(s, cash):
    acts, nc = get_actions(s, cash)
    if len(acts) == 0:
        return -np.inf, None

    ns = s + acts * packs
    exp_v = np.zeros(len(acts))

    for sc in market[2]:
        next_s = ns * np.array(sc['r'])
        exp_v += sc['p'] * solve_stage3_batch(next_s, nc)

    best_i = int(np.argmax(exp_v))
    return float(exp_v[best_i]), tuple(int(x) for x in acts[best_i])


def solve_stage1(s, cash):
    acts, nc = get_actions(s, cash)
    best_val = -np.inf
    best_u = None

    for i in range(len(acts)):
        u = acts[i]
        c_post = nc[i]
        s_post = s + u * packs

        exp_v = 0.0
        for sc in market[1]:
            next_s = s_post * np.array(sc['r'])
            v2, _ = solve_stage2(next_s, c_post)
            exp_v += sc['p'] * v2

        if exp_v > best_val:
            best_val = exp_v
            best_u = tuple(int(x) for x in u)

    return best_val, best_u


def build_tree(s0, cash0, u1_opt):
    outcomes = []
    nodes_info = []
    start_w = float(np.sum(s0) + cash0)

    def walk(stage, s, cash, u, prob, path, ops, comm_sum):
        nodes_info.append({
            'stage': stage,
            'path': " - ".join(path) if path else "Старт",
            's': np.round(s, 2).tolist(),
            'cash': round(cash, 2),
            'u': u
        })

        u_arr = np.array(u)
        step_comm = float(np.sum(np.abs(u_arr) * packs * comm))
        c_post = cash - float(np.sum(u_arr * packs + np.abs(u_arr) * packs * comm))
        s_post = s + u_arr * packs
        cur_ops = ops + [u]

        for sc in market[stage]:
            next_s = s_post * np.array(sc['r'])
            next_p = prob * sc['p']
            next_path = path + [sc['name']]
            next_comm = comm_sum + step_comm

            if stage == 1:
                _, u2 = solve_stage2(next_s, c_post)
                walk(2, next_s, c_post, u2, next_p, next_path, cur_ops, next_comm)
            elif stage == 2:
                _, u3 = solve_stage3_single(next_s, c_post)
                walk(3, next_s, c_post, u3, next_p, next_path, cur_ops, next_comm)
            else:
                final_w = float(np.sum(next_s) + c_post)
                profit = final_w - start_w
                ret_pct = (profit / start_w) * 100.0
                outcomes.append({
                    'path': " - ".join(next_path),
                    'prob': next_p,
                    'ops': cur_ops,
                    'comm': next_comm,
                    'wealth': final_w,
                    'profit': profit,
                    'ret': ret_pct
                })

    walk(1, s0, cash0, u1_opt, 1.0, [], [], 0.0)
    return outcomes, nodes_info


def brute_force_actions(s, cash):
    min_u = [math.ceil((min_s[i] - s[i] - 1e-9) / packs[i]) for i in range(3)]
    max_cash = cash + sum(-min_u[i] * packs[i] * (1 - comm[i]) for i in range(3) if min_u[i] < 0)
    max_u = [int(max_cash // (packs[i] * (1 + comm[i]))) for i in range(3)]
    valid = []
    for u in product(range(min_u[0], max_u[0] + 1), range(min_u[1], max_u[1] + 1), range(min_u[2], max_u[2] + 1)):
        new_s = [s[i] + u[i] * packs[i] for i in range(3)]
        new_cash = cash - sum(u[i] * packs[i] + abs(u[i]) * packs[i] * comm[i] for i in range(3))
        if new_cash >= -1e-9 and all(new_s[i] >= min_s[i] - 1e-9 for i in range(3)):
            valid.append((u, np.array(new_s), new_cash))
    return valid


def brute_force_test():
    small_s = np.array([100.0, 300.0, 200.0])
    small_cash = 60.0

    v_dp, u1_dp = solve_stage1(small_s, small_cash)

    best_bf = -np.inf
    best_u1_bf = None
    for u1, s1_post, c1 in brute_force_actions(small_s, small_cash):
        exp1 = 0.0
        for sc1 in market[1]:
            s2 = s1_post * np.array(sc1['r'])
            best_v2 = -np.inf
            for _, s2_post, c2 in brute_force_actions(s2, c1):
                exp2 = 0.0
                for sc2 in market[2]:
                    s3 = s2_post * np.array(sc2['r'])
                    best_v3 = -np.inf
                    for _, s3_post, c3 in brute_force_actions(s3, c2):
                        exp3 = sum(
                            sc3['p'] * (np.sum(s3_post * np.array(sc3['r'])) + c3)
                            for sc3 in market[3]
                        )
                        if exp3 > best_v3:
                            best_v3 = exp3
                    exp2 += sc2['p'] * best_v3
                if exp2 > best_v2:
                    best_v2 = exp2
            exp1 += sc1['p'] * best_v2
        if exp1 > best_bf:
            best_bf = exp1
            best_u1_bf = u1

    assert abs(v_dp - best_bf) < 1e-4, f"Ошибка сверки значений: DP={v_dp}, BF={best_bf}"
    assert u1_dp == best_u1_bf, f"Ошибка сверки планов: DP={u1_dp}, BF={best_u1_bf}"
    print(f"Сверка с полным перебором: пройдена (V_DP = V_BF = {v_dp:.2f}, u1 = {u1_dp})")


def run_extra_tests():
    print("\nРезультаты тестов:")
    orig_r = {k: [list(sc['r']) for sc in market[k]] for k in market}

    def restore():
        for k in market:
            for i in range(len(market[k])):
                market[k][i]['r'] = list(orig_r[k][i])

    acts_t1, _ = get_actions(min_s, 0.0)
    print(f"Тест 1 (кэш = 0):         Ожид: [[0, 0, 0]]   | Факт: {acts_t1.tolist()}")

    acts_t2, nc_t2 = get_actions(np.array([100.0, 800.0, 400.0]), 100.0, cur_min_s=np.array([30.0, 800.0, 400.0]))
    mask_t2 = (acts_t2[:, 1] == 0) & (acts_t2[:, 2] == 0)
    acts_t2_f = acts_t2[mask_t2]
    nc_t2_f = nc_t2[mask_t2]
    r_t2 = np.sum([sc['p'] * np.array(sc['r']) for sc in market[1]], axis=0)
    vals_t2 = np.sum((np.array([100.0, 800.0, 400.0]) + acts_t2_f * packs) * r_t2, axis=1) + nc_t2_f
    best_t2 = acts_t2_f[int(np.argmax(vals_t2))].tolist()
    print(f"Тест 2 (только ЦБ1):      Ожид: [3, 0, 0]     | Факт: {best_t2}")

    for k in market:
        for sc in market[k]:
            sc['r'] = [1.0, 1.0, 1.0]
    _, u_flat = solve_stage3_single(start_s, start_cash)
    restore()
    print(f"Тест 3 (рост R = 1.0):    Ожид: (0, 0, 0)     | Факт: {u_flat}")

    acts_t4, nc_t4 = get_actions(start_s, 60.0)
    m4 = (acts_t4[:, 1] == 0) & (acts_t4[:, 2] == 0) & (acts_t4[:, 0] >= 0)
    v_t4 = np.sum((start_s + acts_t4[m4] * packs) * np.array([1.04, 1.0, 1.0]), axis=1) + nc_t4[m4]
    m_opt = acts_t4[m4][np.abs(v_t4 - np.max(v_t4)) < 1e-5]
    print(f"Тест 4 (равные планы):    Ожид: 3 плана       | Факт: {len(m_opt)} плана ({m_opt.tolist()})")

    for sc in market[3]:
        sc['r'] = [3.0, 1.0, 1.0]
    _, u_super = solve_stage3_single(start_s, 260.0)
    restore()
    print(f"Тест 5 (сверхдоход ЦБ1):  Ожид: (42, -3, -3)  | Факт: {u_super}")

    acts_neg, _ = get_actions(min_s, -100.0)
    bad_market = {1: [{'name': 'Б', 'p': 0.5, 'r': [1.2, 1.1, 1.07]}], 2: market[2], 3: market[3]}
    try:
        validate_market(bad_market)
        mkt_err = "ошибка не выявлена"
    except ValueError as err:
        mkt_err = f"ValueError ({err})"
    print(f"Тест 6 (некорр. вход):    Ожид: 0 действ., ValueError | Факт: {len(acts_neg)} действ., {mkt_err}")


if __name__ == '__main__':
    validate_market(market)

    t0 = time.perf_counter()
    v1_opt, u1_opt = solve_stage1(start_s, start_cash)
    elapsed = time.perf_counter() - t0

    print(f"Время расчета: {elapsed:.3f} сек")
    print(f"Оптимальный план на 1 этапе: {u1_opt}")
    print(f"Ожидаемый конечный капитал: {v1_opt:.2f} д.е.\n")

    tree, nodes = build_tree(start_s, start_cash, u1_opt)

    print("Узлы принятия решений:")
    for nd in nodes:
        print(f"Этап {nd['stage']} [{nd['path']:<9}]: активы={nd['s']}, кэш={nd['cash']}, u={nd['u']}")

    print("\nДерево исходов:")
    print(f"{'№':<3} {'Сценарий':<11} {'Вероятность':<12} {'Операции (ЦБ1)':<16} {'Капитал':<10} {'Доход':<10} {'Доходность':<11} {'Комиссия':<8}")
    for idx, o in enumerate(tree, 1):
        ops_str = f"({o['ops'][0][0]:+d}; {o['ops'][1][0]:+d}; {o['ops'][2][0]:+d})"
        print(f"{idx:<3} {o['path']:<11} {o['prob']:<12.4f} {ops_str:<16} {o['wealth']:<10.2f} {o['profit']:<+10.2f} {o['ret']:<+8.2f}%   {o['comm']:<8.2f}")

    exp_w = sum(o['wealth'] * o['prob'] for o in tree)
    exp_p = sum(o['profit'] * o['prob'] for o in tree)
    exp_r = sum(o['ret'] * o['prob'] for o in tree)
    prob_loss = sum(o['prob'] for o in tree if o['profit'] < 0)
    min_p = min(o['profit'] for o in tree)
    max_p = max(o['profit'] for o in tree)

    print("\nИтог:")
    print(f"Ожидаемый капитал: {exp_w:.2f} д.е.")
    print(f"Ожидаемый доход: {exp_p:+.2f} д.е. ({exp_r:+.2f}%)")
    print(f"Вероятность убытка: {prob_loss * 100:.1f}%")
    print(f"Максимальный доход: {max_p:+.2f} д.е.")
    print(f"Минимальный доход: {min_p:+.2f} д.е.\n")

    brute_force_test()
    run_extra_tests()