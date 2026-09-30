import numpy as np


#перерасчет таблицы
def step_compact(matrix, b, c_row, q, basis_vars, free_vars):
    rows_count, cols_count = matrix.shape

    #самый большой минус в нижней строке (разрешающий столбец)
    j_star = np.argmin(c_row)

    #самое маленькое отношение (разрешающая строка)
    ratios = [b[i] / matrix[i, j_star] if matrix[i, j_star] > 0 else np.inf for i in range(rows_count)]
    min_ratio = np.min(ratios)

    if min_ratio == np.inf:
        return (None,) * 6

    i_star = np.argmin(ratios)
    pivot = matrix[i_star, j_star]

    matrix_new = np.zeros((rows_count, cols_count))
    b_new = np.zeros(rows_count)
    c_row_new = np.zeros(cols_count)

    #новый разрешающий элемент
    matrix_new[i_star, j_star] = 1.0 / pivot

    #новая разрешающая строка
    for j in range(cols_count):
        if j != j_star:
            matrix_new[i_star, j] = matrix[i_star, j] / pivot
    b_new[i_star] = b[i_star] / pivot

    #новый разрешающий столбец
    for i in range(rows_count):
        if i != i_star:
            matrix_new[i, j_star] = -matrix[i, j_star] / pivot
    c_row_new[j_star] = -c_row[j_star] / pivot

    #перерасчет остальных значений в строках и столбцах
    for i in range(rows_count):
        if i != i_star:
            for j in range(cols_count):
                if j != j_star:
                    matrix_new[i, j] = matrix[i, j] - (matrix[i_star, j] * matrix[i, j_star]) / pivot
            b_new[i] = b[i] - (b[i_star] * matrix[i, j_star]) / pivot

    for j in range(cols_count):
        if j != j_star:
            c_row_new[j] = c_row[j] - (matrix[i_star, j] * c_row[j_star]) / pivot

    #новое значение в правом нижнем углу таблицы
    q_new = q - (b[i_star] * c_row[j_star]) / pivot

    #меняем местами имена базисной и свободной переменной
    basis_vars[i_star], free_vars[j_star] = free_vars[j_star], basis_vars[i_star]

    return matrix_new, b_new, c_row_new, q_new, basis_vars, free_vars


#основное решение
def solve(target_func_text, constraints_list, optimization_mode='min'):
    c_initial = [float(x) for x in target_func_text.split()]
    orig_vars_count = len(c_initial)
    constraints_count = len(constraints_list)

    matrix_rows = []
    b_values = []
    relation_signs = []
    for line in constraints_list:
        parts = line.split()
        b_values.append(float(parts[-1]))
        relation_signs.append(parts[-2])
        matrix_rows.append([float(x) for x in parts[:-2]])

    #если задача на максимум, меняем для решения задачи на минимум
    if optimization_mode.strip().lower() == 'max':
        c_minimized = [-x for x in c_initial]
    else:
        c_minimized = c_initial.copy()

    #приводим к каноническому виду
    slack_variables = []
    for i in range(constraints_count):
        val_b = b_values[i]
        sign = relation_signs[i]
        if val_b < 0:
            matrix_rows[i] = [-x for x in matrix_rows[i]]
            b_values[i] = -val_b
            if sign == '<=':
                sign = '>='
            elif sign == '>=':
                sign = '<='

        if sign == '<=':
            slack_variables.append((i, 1.0))
        elif sign == '>=':
            slack_variables.append((i, -1.0))

    slacks_count = len(slack_variables)
    all_canon_vars_count = orig_vars_count + slacks_count
    matrix_canonical = np.zeros((constraints_count, all_canon_vars_count))

    for i in range(constraints_count):
        matrix_canonical[i, :orig_vars_count] = matrix_rows[i]

    for col_idx, (row_idx, coef) in enumerate(slack_variables):
        matrix_canonical[row_idx, orig_vars_count + col_idx] = coef

    #проверяем, есть ли готовый базис среди исходных переменных
    basis_vars = [None] * constraints_count
    used_columns = set()

    for i in range(constraints_count):
        for j in range(orig_vars_count):
            if j not in used_columns and matrix_canonical[i, j] == 1.0:
                if all(matrix_canonical[k, j] == 0.0 for k in range(constraints_count) if k != i):
                    basis_vars[i] = f"x{j + 1}"
                    used_columns.add(j)
                    break

    #если для строки базиса нет, добавляем туда переменную
    artificial_count = 0
    for i in range(constraints_count):
        if basis_vars[i] is None:
            artificial_count += 1
            basis_vars[i] = f"x{all_canon_vars_count + artificial_count}"

    free_vars = [f"x{j + 1}" for j in range(all_canon_vars_count) if f"x{j + 1}" not in basis_vars]
    free_indices = [int(name[1:]) - 1 for name in free_vars]

    current_matrix = matrix_canonical[:, free_indices].copy()
    current_b = np.array(b_values, dtype=float)

    #решаем вспомогательную задачу
    if artificial_count > 0:
        c_row = np.zeros(len(free_vars))
        q = 0.0
        for i in range(constraints_count):
            if int(basis_vars[i][1:]) > all_canon_vars_count:
                c_row -= current_matrix[i]
                q -= current_b[i]

        while np.min(c_row) < 0:
            current_matrix, current_b, c_row, q, basis_vars, free_vars = step_compact(current_matrix, current_b, c_row, q, basis_vars, free_vars)

            if current_matrix is None:
                print("Область допустимых решений не ограничена")
                return

        if q != 0:
            print("Решений нет")
            return

        #убираем столбцы искусственных переменных
        valid_cols = [idx for idx, name in enumerate(free_vars) if int(name[1:]) <= all_canon_vars_count]
        current_matrix = current_matrix[:, valid_cols]
        free_vars = [free_vars[idx] for idx in valid_cols]

    #переходим к основной задаче и подставляем функцию
    full_c_vector = np.zeros(all_canon_vars_count)
    full_c_vector[:orig_vars_count] = c_minimized

    c_row = np.zeros(len(free_vars))
    q = 0.0

    for j, col_name in enumerate(free_vars):
        var_idx = int(col_name[1:]) - 1
        c_row[j] = full_c_vector[var_idx]

    for i, row_name in enumerate(basis_vars):
        var_idx = int(row_name[1:]) - 1
        if var_idx < all_canon_vars_count:
            coef = full_c_vector[var_idx]
            c_row -= coef * current_matrix[i]
            q -= coef * current_b[i]

    while np.min(c_row) < 0:
        current_matrix, current_b, c_row, q, basis_vars, free_vars = step_compact(current_matrix, current_b, c_row, q, basis_vars, free_vars)

        if current_matrix is None:
            print("Область допустимых решений не ограничена")
            return

    #выводим значения только для исходных переменных
    solution_dict = {f"x{i + 1}": 0.0 for i in range(orig_vars_count)}
    for i, name in enumerate(basis_vars):
        if name in solution_dict:
            solution_dict[name] = float(current_b[i])

    solution_vector = [solution_dict[f"x{i + 1}"] for i in range(orig_vars_count)]
    optimal_function_value = float(np.dot(c_initial, solution_vector))

    for name, val in solution_dict.items():
        print(f"  {name} = {round(val, 4)}")
    point_string = ", ".join(str(round(val, 4)) for val in solution_vector)
    print(f"Оптимальная точка: x* = ({point_string})")
    print(f"Значение функции:  W(x*) = {round(optimal_function_value, 4)}")


if __name__ == "__main__":
    user_func = input("Коэффициенты целевой функции (например: 2 3 1 4): ")
    user_mode = input("Направление оптимизации (min или max): ")

    constraints_total = int(input("Количество ограничений (например: 3): "))
    print("Введите ограничения построчно (например: 1 2 1 0 <= 7):")

    user_constraints = []
    for i in range(constraints_total):
        user_line = input(f"Ограничение {i + 1}: ")
        user_constraints.append(user_line)

    solve(user_func, user_constraints, user_mode)
