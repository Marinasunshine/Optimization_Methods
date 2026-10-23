import sympy as sp

eps = 1e-8
x, y, z = sp.symbols("x y z", real=True)
lam1, lam2 = sp.symbols("lambda1 lambda2", real=True)

variables = [x, y, z]
multipliers = [lam1, lam2]

f = x ** 2 + y ** 2 + z ** 2
constraints = [x + y + z - 3, x - y - 1]

L = f + sum(a * h for a, h in zip(multipliers, constraints))

grad_L = [sp.diff(L, v) for v in variables]
equations = grad_L + constraints

print("Целевая функция:", f)
print("Ограничения:", [f"{h} = 0" for h in constraints])
print("Функция Лагранжа:", L)
print("Уравнения стационарности:")
for eq in grad_L:
    print(f"  {eq} = 0")


unknowns = variables + multipliers
raw_solutions = sp.solve(equations, unknowns, dict=True)

solutions = []
for sol in raw_solutions:
    if all(val.is_real for val in sol.values()):
        solutions.append(sol)

candidates = []
for number, solution in enumerate(solutions, start=1):
    point_subs = {v: solution[v] for v in variables}
    mult_subs = {m: solution[m] for m in multipliers}

    residuals = [sp.simplify(h.subs(point_subs)) for h in constraints]
    is_admissible = all(abs(float(r)) <= eps for r in residuals)

    value = sp.simplify(f.subs(point_subs))

    candidates.append({"point": point_subs, "multipliers": mult_subs, "residuals": residuals, "value": value, "admissible": is_admissible})

    print(f"Кандидат {number}:")
    print(f"точка: ({point_subs[x]}, {point_subs[y]}, {point_subs[z]})")
    print(f"множители: lambda1 = {mult_subs[lam1]}, lambda2 = {mult_subs[lam2]}")
    print(f"значение функции: {value} (численно: {float(value):.4f})")
    print(f"невязки ограничений: {[float(r) for r in residuals]}")
    print(f"допустимость: {'да' if is_admissible else 'нет'}")
    print()

if candidates:
    best_min = min(candidates, key=lambda c: float(c["value"]))
    pt_str = f"({best_min['point'][x]}, {best_min['point'][y]}, {best_min['point'][z]})"

    param_sol = sp.solve(constraints, [y, z], dict=True)

    if param_sol:
        y_param = param_sol[0][y]
        z_param = param_sol[0][z]
        f_param = sp.simplify(f.subs({y: y_param, z: z_param}))

        lim_plus_inf = sp.limit(f_param, x, sp.oo)
        lim_minus_inf = sp.limit(f_param, x, -sp.oo)

        if lim_plus_inf == sp.oo and lim_minus_inf == sp.oo:
            print(f"глобальный минимум: {pt_str}, f = {best_min['value']}")
        else:
            print("глобальный минимум: отсутствует")

        if lim_plus_inf == -sp.oo and lim_minus_inf == -sp.oo:
            best_max = max(candidates, key=lambda c: float(c["value"]))
            pt_str_max = f"({best_max['point'][x]}, {best_max['point'][y]}, {best_max['point'][z]})"
            print(f"глобальный максимум: {pt_str_max}, f = {best_max['value']}")
        else:
            print("глобальный максимум: отсутствует")

        print("основание классификации:")
        print(f"  Ограничения использованы для параметризации: y = {y_param}, z = {z_param}.")
        print(f"  Сужение функции на множество: f(x) = {f_param}.")
        print(f"  Анализ пределов: lim(x->+oo) f(x) = {lim_plus_inf}, lim(x->-oo) f(x) = {lim_minus_inf}.")
        if lim_plus_inf == sp.oo and lim_minus_inf == sp.oo:
            print("  Функция коэрцитивна (уходит в +oo), поэтому существует только глобальный минимум.")
    else:
        print("основание классификации: Не удалось выполнить параметризацию.")
