import math
import sys
sys.stdout.reconfigure(encoding='utf-8')

def f1(x):
    """Підінтегральна функція для I1"""
    return (math.exp(2 * x) * math.sin(x)) / (x**2 + 1)

def f2(x):
    """Підінтегральна функція для I2"""
    coth_x = 1.0 / math.tanh(x)
    return (x * coth_x) / (x**3 + 2 * x + 1)


nodes_weights = [
    (0.991455371120813, 0.0,               0.022935322010529),
    (0.949107912342759, 0.129484966168870, 0.063092092629979),
    (0.864864423359769, 0.0,               0.104790010322250),
    (0.741531185599394, 0.279705391489277, 0.140653259715525),
    (0.586087235467691, 0.0,               0.169004726639267),
    (0.405845151377397, 0.381830050505119, 0.190350578064785),
    (0.207784955007898, 0.0,               0.204432940075298),
    (0.000000000000000, 0.417959183673469, 0.209482141084728)
]

def g7_k15(f, a, b):
    """Обчислення G7 та K15, оцінка похибки."""
    mid = 0.5 * (a + b)
    half_diff = 0.5 * (b - a)
    
    G7 = 0.0
    K15 = 0.0
    
    for t, w_g, w_k in nodes_weights:
        if t == 0.0:
            val = f(mid)
            G7 += w_g * val
            K15 += w_k * val
        else:
            val = f(mid + half_diff * t) + f(mid - half_diff * t)
            G7 += w_g * val
            K15 += w_k * val
            
    G7 *= half_diff
    K15 *= half_diff
    
    error = (200.0 * abs(G7 - K15)) ** 1.5
    return K15, error

def adaptive_algorithm(f, a, b, tol):
    stack = [(a, b, tol)]
    
    total_integral = 0.0
    total_error = 0.0
    
    while stack:
        curr_a, curr_b, curr_tol = stack.pop()
        val, err = g7_k15(f, curr_a, curr_b)
        
        if err < curr_tol:
            # Точність досягнута
            total_integral += val
            total_error += err
        else:
            # Ділимо відрізок навпіл
            mid = 0.5 * (curr_a + curr_b)
            
            stack.append((mid, curr_b, curr_tol / 2.0))
            stack.append((curr_a, mid, curr_tol / 2.0))
            
    return total_integral, total_error


if __name__ == "__main__":
    tolerances = [1e-3, 1e-6]

    print("="*40)
    print("Пара Гауса-Кронрода: Інтеграл I1")
    print("="*40)
    for tol in tolerances:
        result, error = adaptive_algorithm(f1, 0, 2, tol)
        print(f"Задана точність: {tol}")
        print(f"Результат (K15): {result:.10f}")
        print(f"Похибка:         {error:.4e}\n")
    
    print("="*40)
    print("Пара Гауса-Кронрода: Інтеграл I2")
    print("="*40)
    for tol in tolerances:
        result, error = adaptive_algorithm(f2, 1, 2, tol)
        print(f"Задана точність: {tol}")
        print(f"Результат (K15): {result:.10f}")
        print(f"Похибка:         {error:.4e}\n")



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