import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f(x, y):
    return x * y + np.exp(0.1 * x)
def df_dy(x, y):
    return x


def rk4(f, x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h / 2.0, y + k1 / 2.0)
    k3 = h * f(x + h / 2.0, y + k2 / 2.0)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

def explicit_adams_4(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
            
    x[0] = x0
    y[0] = y0
            
    # Обчислюємо 3 перших точки методом Рунге-Кутти 4-го порядку точності
    for i in range(min(3, n)):
        y[i+1] = rk4(f, x[i], y[i], h)
        x[i+1] = x[0] + (i + 1) * h
                
    # Явний метод Адамса 4-го порядку
    for i in range(3, n):
        x[i+1] = x[0] + (i + 1) * h

        f_j1 = f(x[i], y[i])         # f_{j-1}
        f_j2 = f(x[i-1], y[i-1])     # f_{j-2}
        f_j3 = f(x[i-2], y[i-2])     # f_{j-3}
        f_j4 = f(x[i-3], y[i-3])     # f_{j-4}
            
        # sum(b_i)=1
        y[i+1] = y[i] + (h / 24.0) * (55 * f_j1 - 59 * f_j2 + 37 * f_j3 - 9 * f_j4)
        
    return x, y

def implicit_adams_4_iteration(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)

    x[0] = x0
    y[0] = y0

    # Знаходимо y1, y2, y3 за допомогою методу Рунге-Кутти 4-го порядку точності
    for i in range(min(3, n)):
        y[i+1] = rk4(f, x[i], y[i], h)
        x[i+1] = x[0] + (i + 1) * h

    # Неявний метод Адамса 4-го порядку
    for j in range(3, n):  
        x[j+1] = x[0] + (j + 1) * h

        # Значення функції у попередніх вузлах
        f_j1 = f(x[j], y[j])         # f_{j-1}
        f_j2 = f(x[j-1], y[j-1])     # f_{j-2}
        f_j3 = f(x[j-2], y[j-2])     # f_{j-3}
        f_j4 = f(x[j-3], y[j-3])     # f_{j-4}

        # Предиктор (Початкове наближення y_j^(0))
        # Використовуємо явний метод Адамса 4-го порядку
        y_predict = y[j] + (h / 24.0) * (55 * f_j1 - 59 * f_j2 + 37 * f_j3 - 9 * f_j4)
        
        # Коректор (Метод простої ітерації)
        for k in range(1000000): # k=0 => метод предиктор-коректор
            f_predict = f(x[j+1], y_predict) # f(x_j, y_j^(k))
            # sum(b_i)=1
            y_correct = y[j] + (h / 24.0) * (9.0 * f_predict + 19.0 * f_j1 - 5.0 * f_j2 + f_j3)

            # Перевірка на збіжність
            if abs(y_correct - y_predict) < 1e-11:
                break
                
            y_predict = y_correct
                
        y[j+1] = y_predict

    return x, y

def implicit_adams_4_Newton(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)

    x[0] = x0
    y[0] = y0

    for i in range(min(3, n)):
        y[i+1] = rk4(f, x[i], y[i], h)
        x[i+1] = x[0] + (i + 1) * h

    # Неявний метод Адамса 4-го порядку
    for j in range(3, n):  
        x[j+1] = x[0] + (j + 1) * h

        f_j1 = f(x[j], y[j])         # f_{j-1}
        f_j2 = f(x[j-1], y[j-1])     # f_{j-2}
        f_j3 = f(x[j-2], y[j-2])     # f_{j-3}
        f_j4 = f(x[j-3], y[j-3])     # f_{j-4}

        # Предиктор (Початкове наближення y_j^(0))
        # Використовуємо явний метод Адамса 4-го порядку
        y_predict = y[j] + (h / 24.0) * (55 * f_j1 - 59 * f_j2 + 37 * f_j3 - 9 * f_j4)

        # Частина формули коректора, яка не залежить від невідомого y_{j+1}
        const_term = y[j] + (h / 24.0) * (19.0 * f_j1 - 5.0 * f_j2 + f_j3)
        
        # Коректор (Метод Ньютона)
        for _ in range(1000000): # захист від нескінченного циклу
            # Значення F(y)
            F_y = y_predict - const_term - (h / 24.0) * 9.0 * f(x[j+1], y_predict)
            
            # Значення похідної F'(y)
            dF_dy = 1.0 - (3.0 * h / 8.0) * df_dy(x[j+1], y_predict)
            
            # Крок Ньютона
            y_correct = y_predict - (F_y / dF_dy)
            
            # Перевірка на збіжність
            if abs(y_correct - y_predict) < 1e-11:
                y_predict = y_correct
                break
                
            y_predict = y_correct
                
        y[j+1] = y_predict

    return x, y

def get_exact_curve(x_start, x_end, y_start):
    sol = solve_ivp(f, [x_start, x_end], [y_start], dense_output=True, rtol=1e-11, atol=1e-11)
    
    x_smooth = np.linspace(x_start, x_end, 500)
    y_smooth = sol.sol(x_smooth)[0]
    return x_smooth, y_smooth

if __name__ == "__main__":
    x_start = 0.0
    x_end = 1.0
    y_start = 1.0

    n = [4, 8, 16]

    for i in n:
        step = 1.0 / i
        num_steps = i
        x_vals, y_vals = explicit_adams_4(f, x_start, y_start, step, num_steps)
        
        plt.figure(figsize=(8, 6))
        x_exact, y_exact = get_exact_curve(x_start, x_end, y_start)
        plt.plot(x_exact, y_exact, color='red', label="Точний розв'язок")
        plt.plot(x_vals, y_vals, marker='o', linestyle='', color='blue', label="Чисельний розв'язок")

        plt.title(f"Розв'язок явного метода Адамса при n = {i} (h = {step})", fontsize=12)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.show()

    for i in n:
        step = 1.0 / i
        num_steps = i
        x_vals, y_vals = implicit_adams_4_iteration(f, x_start, y_start, step, num_steps)
        
        plt.figure(figsize=(8, 6))
        x_exact, y_exact = get_exact_curve(x_start, x_end, y_start)
        plt.plot(x_exact, y_exact, color='red', label="Точний розв'язок")
        plt.plot(x_vals, y_vals, marker='o', linestyle='', color='black', label="Чисельний розв'язок")

        plt.title(f"Неявний метод Адамса (метод простої ітерації) при n = {i} (h = {step})", fontsize=12)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.show()

    for i in n:
        step = 1.0 / i
        num_steps = i
        x_vals, y_vals = implicit_adams_4_Newton(f, x_start, y_start, step, num_steps)

        plt.figure(figsize=(8, 6))
        x_exact, y_exact = get_exact_curve(x_start, x_end, y_start)
        plt.plot(x_exact, y_exact, color='red', label="Точний розв'язок")
        plt.plot(x_vals, y_vals, marker='o', linestyle='', color='green', label="Чисельний розв'язок")

        plt.title(f"Неявний метод Адамса (метод Ньютона) при n = {i} (h = {step})", fontsize=12)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.show()









# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp

# def f(x, y):
#     return x * y + np.exp(0.1 * x)

# # Метод Рунге-Кутти 4-го порядку
# def rk4(f, x, y, h):
#     k1 = h * f(x, y)
#     k2 = h * f(x + h / 2.0, y + k1 / 2.0)
#     k3 = h * f(x + h / 2.0, y + k2 / 2.0)
#     k4 = h * f(x + h, y + k3)
#     return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

# # Метод Ейлера (метод ламаних)
# def euler_method(f, x, y, h):        
#     return y + h * f(x, y)

# # Метод Хойна
# def heun_method(f, x, y, h):
#     k1 = h * f(x, y)
#     k2 = h * f(x + h, y + k1)
#     return y + (k1 + k2) / 2.0

# # Метод середньої точки
# def midpoint_method(f, x, y, h):
#     k1 = h * f(x, y)
#     k2 = h * f(x + h / 2.0, y + k1 / 2.0)
#     return y + k2

# # Апостеріорний контроль похибки (адаптивний крок)
# def a_posteriori_error_estimation(f, x0, y0, x_end, method, s, h0, eps0, M=1.0):
#     # s = 1 # Порядок методу Ейлера
#     # s = 2 # Порядок методу Хойна і середньої точки
#     # s = 4  # Порядок методу Рунге-Кутти

#     eps1 = eps0 / (2**(s + 1))
    
#     x_vals = [x0]
#     y_vals = [y0]
#     h_vals = []
    
#     x = x0
#     y = y0
#     h = h0
    
#     while x < x_end:
#         h_step = min(h, x_end - x)
            
#         while True:            
#             # y_tilde_2: результат за один крок довжини h
#             y_tilde_2 = method(f, x, y, h_step)
            
#             # y_2: результат за два кроки довжини h
#             y_mid = method(f, x, y, h_step / 2.0)
#             y_2 = method(f, x + h_step / 2.0, y_mid, h_step / 2.0)
            
#             # Головний член похибки обчислення у(х+2h)
#             r = (y_2 - y_tilde_2) / (2**s - 1)
            
#             # Апостеріорна оцінка похибки psi_i(h)
#             psi = abs(r) / max(M, abs(y))
            
#             if psi > eps0:
#                 # Точність незадовільна: зменшуємо крок удвічі та повторюємо спробу
#                 h /= 2.0
#                 h_step = min(h, x_end - x)
#             else:
#                 # Точність задовільна: приймаємо крок
#                 # Використовуємо уточнене значення
#                 y_next = y_2 + r 
#                 break
                
#         x += h_step
#         y = y_next
        
#         x_vals.append(x)
#         y_vals.append(y)
#         h_vals.append(h_step)
        
#         # Управління розміром кроку для наступної ітерації
#         if psi < eps1 and h_step == h:
#             h *= 2.0  # Якщо похибка дуже мала, збільшуємо крок
#         # Якщо eps1 <= psi <= eps0, наступний крок залишається рівним H
        
#     return np.array(x_vals), np.array(y_vals), np.array(h_vals)

# def get_smooth_curve(f, x0, y0, x_end):    
#     sol = solve_ivp(
#         fun=lambda t, y: f(t, y), 
#         t_span=(x0, x_end), 
#         y0=[y0], 
#         method='DOP853',       # метод Рунге-Кутти 8-го порядку
#         dense_output=True,     # створює інтерполянт для отримання плавної кривої
#         rtol=1e-11,            # висока відносна точність
#         atol=1e-11             # висока абсолютна точність
#     )
    
#     x_smooth = np.linspace(x0, x_end, 500)
#     y_smooth = sol.sol(x_smooth)[0]
#     return x_smooth, y_smooth

# def solve_fixed_step(f, step_method, x0, y0, x_end, h0):
#     N = int(np.round((x_end - x0) / h0))
#     h = (x_end - x0) / N
    
#     x_vals = np.linspace(x0, x_end, N + 1)
#     y_vals = np.zeros(N + 1)
    
#     y_vals[0] = y0
#     y = y0
    
#     for i in range(1, N + 1):
#         y = step_method(f, x_vals[i-1], y, h)
#         y_vals[i] = y
        
#     return x_vals, y_vals

# def plot_results(x_rk4, y_rk4, x_res, y_res, f, x0, y0, x_end, title1, title2):
#     x_exact, y_exact = get_smooth_curve(f, x0, y0, x_end)

#     plt.figure(figsize=(8, 6))
#     plt.plot(x_exact, y_exact, 'r-', label="Точний розв'язок")
#     plt.scatter(x_rk4, y_rk4, label="Числовий розв'язок", zorder=5)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title(title1)
#     plt.grid(True)
#     plt.legend()
#     plt.show()


#     plt.figure(figsize=(8, 6))
#     plt.plot(x_exact, y_exact, 'r-', label="Точний розв'язок")
#     plt.scatter(x_res, y_res, label="Числовий розв'язок", zorder=5)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title(title2)
#     plt.grid(True)
#     plt.legend()
#     plt.show()


# if __name__ == "__main__":
#     x0 = 0.0
#     y0 = 1.0
#     x_end = 1.0
#     h0 = 1e-13
#     epsilon = 1e-10
#     M = 1.0 # Параметр розмірності

#     methods = [
#         ("РК4", "Рунге-Кутти 4-го порядку точності", rk4, 4),
#         ("Ейлера", "Ейлера", euler_method, 1),
#         ("Хойна", "Хойна", heun_method, 2),
#         ("середньої точки", "середньої точки", midpoint_method, 2)
#     ]

#     results_fixed = {}
#     results_aposteriori = {}

#     # --- Блок 1: Розв'язок з фіксованим кроком ---
#     for short_name, full_name, func, order in methods:
#         print("=" * 70)
#         print(f"Розв'язок методом {short_name}:")
#         print("-" * 40)
#         print(f"{'x':<10} | {'y(x)':<18}")
#         print("-" * 40)
        
#         x_res, y_res = solve_fixed_step(f, func, x0, y0, x_end, 0.1)
#         results_fixed[short_name] = (x_res, y_res)

#         for x_val, y_val in zip(x_res, y_res):
#             print(f"{x_val:<10.6f} | {y_val:<18.8f}")
            
#         print("-" * 40)
#         print(f"Загальна кількість обчислених точок: {len(x_res)}\n")


#     print("\n" + "=" * 70)
#     print("=" * 70 + "\n")


#     # --- Блок 2: Розв'язок з апостеріорним контролем похибки ---
#     for short_name, full_name, func, order in methods:
#         print("=" * 70)
#         print(f"Розв'язок методом {short_name} з апостеріорним контролем похибки (eps0 = {epsilon}):")
#         print("-" * 70)
#         print(f"{'x':<10} | {'y(x)':<18} | {'Крок H, з яким знайдено точку'}")
#         print("-" * 70)
        
#         x_res, y_res, h_res = a_posteriori_error_estimation(f, x0, y0, x_end, func, order, h0, eps0=epsilon, M=M)
#         results_aposteriori[short_name] = (x_res, y_res)

#         print(f"{x_res[0]:<10.6f} | {y_res[0]:<18.8f} | -")
        
#         for x_val, y_val, h_val in zip(x_res[1:], y_res[1:], h_res):
#             print(f"{x_val:<10.6f} | {y_val:<18.8f} | {h_val:.6f}")
            
#         print("-" * 70)
#         print(f"Загальна кількість обчислених точок: {len(x_res)}\n")


#     for short_name, full_name, _, _ in methods:
#         x_fixed, y_fixed = results_fixed[short_name]
#         x_apost, y_apost = results_aposteriori[short_name]
        
#         plot_results(
#             x_fixed, y_fixed, 
#             x_apost, y_apost, 
#             f, x0, y0, x_end, 
#             f'Метод {full_name}', 
#             f'Метод {full_name} з апостеріорним контролем похибки'
#         )














# ===================================================================
# import math
# import sys
# sys.stdout.reconfigure(encoding='utf-8')

# def f1(x):
#     """Підінтегральна функція для I1"""
#     return (math.exp(2 * x) * math.sin(x)) / (x**2 + 1)

# def f2(x):
#     """Підінтегральна функція для I2"""
#     coth_x = 1.0 / math.tanh(x)
#     return (x * coth_x) / (x**3 + 2 * x + 1)


# nodes_weights = [
#     (0.991455371120813, 0.0,               0.022935322010529),
#     (0.949107912342759, 0.129484966168870, 0.063092092629979),
#     (0.864864423359769, 0.0,               0.104790010322250),
#     (0.741531185599394, 0.279705391489277, 0.140653259715525),
#     (0.586087235467691, 0.0,               0.169004726639267),
#     (0.405845151377397, 0.381830050505119, 0.190350578064785),
#     (0.207784955007898, 0.0,               0.204432940075298),
#     (0.000000000000000, 0.417959183673469, 0.209482141084728)
# ]

# def g7_k15(f, a, b):
#     """Обчислення G7 та K15, оцінка похибки."""
#     mid = 0.5 * (a + b)
#     half_diff = 0.5 * (b - a)
    
#     G7 = 0.0
#     K15 = 0.0
    
#     for t, w_g, w_k in nodes_weights:
#         if t == 0.0:
#             val = f(mid)
#             G7 += w_g * val
#             K15 += w_k * val
#         else:
#             val = f(mid + half_diff * t) + f(mid - half_diff * t)
#             G7 += w_g * val
#             K15 += w_k * val
            
#     G7 *= half_diff
#     K15 *= half_diff
    
#     error = (200.0 * abs(G7 - K15)) ** 1.5
#     return K15, error

# def adaptive_algorithm(f, a, b, tol):
#     stack = [(a, b, tol)]
    
#     total_integral = 0.0
#     total_error = 0.0
    
#     while stack:
#         curr_a, curr_b, curr_tol = stack.pop()
#         val, err = g7_k15(f, curr_a, curr_b)
        
#         if err < curr_tol:
#             # Точність досягнута
#             total_integral += val
#             total_error += err
#         else:
#             # Ділимо відрізок навпіл
#             mid = 0.5 * (curr_a + curr_b)
            
#             stack.append((mid, curr_b, curr_tol / 2.0))
#             stack.append((curr_a, mid, curr_tol / 2.0))
            
#     return total_integral, total_error


# if __name__ == "__main__":
#     tolerances = [1e-3, 1e-6]

#     print("="*40)
#     print("Пара Гауса-Кронрода: Інтеграл I1")
#     print("="*40)
#     for tol in tolerances:
#         result, error = adaptive_algorithm(f1, 0, 2, tol)
#         print(f"Задана точність: {tol}")
#         print(f"Результат (K15): {result:.10f}")
#         print(f"Похибка:         {error:.4e}\n")
    
#     print("="*40)
#     print("Пара Гауса-Кронрода: Інтеграл I2")
#     print("="*40)
#     for tol in tolerances:
#         result, error = adaptive_algorithm(f2, 1, 2, tol)
#         print(f"Задана точність: {tol}")
#         print(f"Результат (K15): {result:.10f}")
#         print(f"Похибка:         {error:.4e}\n")
# ===================================================================

# ===================================================================
# def calculate_S9_S17_runge(f, a, b, num_nodes1, num_nodes2):
#     p = 4

#     segment1 = num_nodes1 - 1
#     val_S1 = simpson_rule(f, a, b, segment1)
    
#     segment2 = num_nodes2 - 1
#     val_S2 = simpson_rule(f, a, b, segment2)
    
#     error = abs(val_S2 - val_S1) / (2**p - 1)
    
#     return val_S1, val_S2, error


# if __name__ == "__main__":
#     # S9:  9 вузлів = 8 відрізків
#     # S17: 17 вузлів = 16 відрізків

#     print("="*50)
#     print("Пара квадратурних формул Сімпсона: Інтеграл I1")
#     print("="*50)
#     val11, val12, err1 = calculate_S9_S17_runge(f1, 0.0, 2.0, 9, 17)
#     print(f"S9  (8 відрізків, 9 точок):   {val11:.8f}")
#     print(f"S17 (16 відрізків, 17 точок): {val12:.8f}")
#     print(f"Оцінка похибки Рунге:         {err1:.2e}\n")

#     print("="*50)
#     print("Пара квадратурних формул Сімпсона: Інтеграл I2")
#     print("="*50)
#     val21, val22, err2 = calculate_S9_S17_runge(f2, 1.0, 2.0, 9, 17)
#     print(f"S9  (8 відрізків, 9 точок):   {val21:.8f}")
#     print(f"S17 (16 відрізків, 17 точок): {val22:.8f}")
#     print(f"Оцінка похибки Рунге:         {err2:.2e}\n")
# ===================================================================

# ===================================================================
# def calculate_S8_S15_runge(f, a, b, num_nodes1, num_nodes2):
#     p = 2

#     segment1 = num_nodes1 - 1
#     val_S1 = trapezoidal_rule(f, a, b, segment1)
    
#     segment2 = num_nodes2 - 1
#     val_S2 = trapezoidal_rule(f, a, b, segment2)
    
#     error = abs(val_S2 - val_S1) / (2**p - 1)
    
#     return val_S1, val_S2, error
    

# if __name__ == "__main__":
#     # S8: 8 вузлів = 7 відрізків
#     # S15: 15 вузлів = 14 відрізків

#     print("="*50)
#     print("Пара квадратурних формул трапеції: Інтеграл I1")
#     print("="*50)
#     val11, val12, err = calculate_S8_S15_runge(f1, 0.0, 2.0, 8, 15)
#     print(f"S8  (7 відрізків, 8 точок):   {val11:.8f}")
#     print(f"S15 (14 відрізків, 15 точок): {val12:.8f}")
#     print(f"Оцінка похибки Рунге:         {err:.2e}\n")

#     print("="*50)
#     print("Пара квадратурних формул трапеції: Інтеграл I2")
#     print("="*50)
#     val21, val22, err = calculate_S8_S15_runge(f2, 1.0, 2.0, 8, 15)
#     print(f"S8  (7 відрізків, 8 точок):   {val21:.8f}")
#     print(f"S15 (14 відрізків, 15 точок): {val22:.8f}")
#     print(f"Оцінка похибки Рунге:         {err:.2e}\n")
# ===================================================================

# ===================================================================
# def simpson_rule(f, a, b, n):
#     """
#     Обчислюємо інтеграл складеною формулою Сімпсона.
#     f: функція, a: нижня межа, b: верхня межа, n: кількість розбиттів (має бути парним)
#     """
#     h = (b - a) / n
#     integral_sum = f(a) + f(b)
    
#     for i in range(1, n):
#         x_i = a + i * h
#         # Парні індекси множимо на 2, непарні - на 4
#         if i % 2 == 0:
#             integral_sum += 2 * f(x_i)
#         else:
#             integral_sum += 4 * f(x_i)
            
#     return integral_sum * (h / 3.0)

# def integrate_runge_simpson(f, a, b, eps):
#     """
#     Обчислюємо інтеграл з заданою точністю eps, використовуючи метод Сімпсона та правило Рунге.
#     """
#     n = 2 # Початкова кількість розбиттів (парне число)
#     p = 4 # Порядок точності методу Сімпсона
    
#     I_n = simpson_rule(f, a, b, n)
    
#     while True:
#         n *= 2
#         I_2n = simpson_rule(f, a, b, n)
#         error = abs(I_2n - I_n) / (2**p - 1)
#         if error < eps:
#             return I_2n, n, error
#         I_n = I_2n


# if __name__ == "__main__":
#     tolerances = [1e-3, 1e-6]
    
#     print("="*45)
#     print("Метод Сімпсона: Інтеграл I1")
#     print("="*45)
#     for eps in tolerances:
#         val, n, err = integrate_runge_simpson(f1, 0.0, 2.0, eps)
#         print(f"Задана точність: {eps}")
#         print(f"Результат:       {val:.8f}")
#         print(f"К-ть розбиттів:  {n}")
#         print(f"Оцінка похибки:  {err:.2e}\n")

#     print("="*45)
#     print("Метод Сімпсона: Інтеграл I2")
#     print("="*45)
#     for eps in tolerances:
#         val, n, err = integrate_runge_simpson(f2, 1.0, 2.0, eps)
#         print(f"Задана точність: {eps}")
#         print(f"Результат:       {val:.8f}")
#         print(f"К-ть розбиттів:  {n}")
#         print(f"Оцінка похибки:  {err:.2e}\n")
# ===================================================================

# ===================================================================
# def trapezoidal_rule(f, a, b, n):
#     """
#     Обчислюємо інтеграл складеною формулою трапецій.
#     f: функція, a: нижня межа, b: верхня межа, n: кількість розбиттів
#     """
#     h = (b - a) / n
#     # Сума значень на кінцях відрізка, поділена на 2
#     integral_sum = 0.5 * (f(a) + f(b))
    
#     # Додаємо значення у внутрішніх вузлах
#     for i in range(1, n):
#         x_i = a + i * h
#         integral_sum += f(x_i)
        
#     return integral_sum * h

# def integrate_runge_trapezoidal(f, a, b, eps):
#     """
#     Обчислюємо інтеграл з заданою точністю eps, використовуючи метод трапецій та правило Рунге.
#     """
#     n = 2
#     p = 2 # Порядок точності методу трапецій
    
#     I_n = trapezoidal_rule(f, a, b, n)
#     while True:
#         n *= 2
#         I_2n = trapezoidal_rule(f, a, b, n)
#         error = abs(I_2n - I_n) / (2**p - 1)
#         if error < eps:
#             return I_2n, n, error
#         I_n = I_2n


# if __name__ == "__main__":
#     tolerances = [1e-3, 1e-6]
    
#     print("="*45)
#     print("Метод трапецій: Інтеграл I1")
#     print("="*45)
#     for eps in tolerances:
#         val, n, err = integrate_runge_trapezoidal(f1, 0.0, 2.0, eps)
#         print(f"Задана точність: {eps}")
#         print(f"Результат:       {val:.8f}")
#         print(f"К-ть розбиттів:  {n}")
#         print(f"Оцінка похибки:  {err:.2e}\n")

#     print("="*45)
#     print("Метод трапецій: Інтеграл I2")
#     print("="*45)
#     for eps in tolerances:
#         val, n, err = integrate_runge_trapezoidal(f2, 1.0, 2.0, eps)
#         print(f"Задана точність: {eps}")
#         print(f"Результат:       {val:.8f}")
#         print(f"К-ть розбиттів:  {n}")
#         print(f"Оцінка похибки:  {err:.2e}\n")
# ===================================================================

# ===================================================================
# def midpoint_rule(f, a, b, n):
#     """
#     Обчислюємо інтеграл складеною формулою середніх прямокутників.
#     f: функція, a: нижня межа, b: верхня межа, n: кількість розбиттів
#     """
#     h = (b - a) / n
#     integral_sum = 0.0
#     for i in range(n):
#         # Точка посередині кожного відрізка
#         x_mid = a + (i + 0.5) * h
#         integral_sum += f(x_mid)
#     return integral_sum * h

# def integrate_runge(f, a, b, eps):
#     """
#     Обчислюємо інтеграл з заданою точністю eps, використовуючи правило Рунге.
#     """
#     n = 2 # Початкова кількість розбиттів
#     p = 2 # Порядок точності методу середніх прямокутників
    
#     # Обчислюємо початкове значення для n розбиттів
#     I_n = midpoint_rule(f, a, b, n)
    
#     while True:
#         n *= 2 # Подвоюємо кількість кроків
#         I_2n = midpoint_rule(f, a, b, n)
        
#         # Оцінка похибки за правилом Рунге
#         error = abs(I_2n - I_n) / (2**p - 1)
        
#         # Якщо досягнуто заданої точності, повертаємо результат
#         if error < eps:
#             return I_2n, n, error
        
#         # Інакше зберігаємо нове значення для наступної ітерації
#         I_n = I_2n


# def main():
#     tolerances = [1e-3, 1e-6]
    
#     print("="*40)
#     print("Обчислення інтеграла I1")
#     print("="*40)
#     for eps in tolerances:
#         val, n, err = integrate_runge(f1, 0.0, 2.0, eps)
#         print(f"Задана точність: {eps}")
#         print(f"Результат:       {val:.8f}")
#         print(f"К-ть розбиттів:  {n}")
#         print(f"Оцінка похибки:  {err:.2e}\n")

#     print("="*40)
#     print("Обчислення інтеграла I2")
#     print("="*40)
#     for eps in tolerances:
#         val, n, err = integrate_runge(f2, 1.0, 2.0, eps)
#         print(f"Задана точність: {eps}")
#         print(f"Результат:       {val:.8f}")
#         print(f"К-ть розбиттів:  {n}")
#         print(f"Оцінка похибки:  {err:.2e}\n")
# if __name__ == "__main__":
#     main()
# ===================================================================