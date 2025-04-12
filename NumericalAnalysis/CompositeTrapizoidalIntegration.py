import math


def function1(x):
    return math.exp((-(x**2) / 2))


def derfunction1(x):
    return -x * math.exp((-(x**2) / 2))


def function2(x):
    return 1 / (2 + math.sin(x))


def derfunction2(x):
    return -1 * (math.cos(x) / ((2 + math.sin(x)) ** 2))


def compTrapQ1(n, a, b):
    h = (b - a) / n
    interior_sum = 0
    for i in range(1, n):
        x_i = a + i * h
        interior_sum += function1(x_i)
    interior_sum *= h
    return (
        (h / 2) * (function1(a))
        + interior_sum
        + (h / 2) * (function1(b))
        + (h**2 / 12) * (derfunction1(a) - derfunction1(b))
    )


def compTrapQ2(n, a, b):
    h = (b - a) / n
    interior_sum = 0
    for i in range(1, n):
        x_i = a + i * h
        interior_sum += function2(x_i)
    interior_sum *= h
    return (
        (h / 2) * (function2(a))
        + interior_sum
        + (h / 2) * (function2(b))
        + (h**2 / 12) * (derfunction2(a) - derfunction2(b))
    )


print("Part A, n=128: " + str(compTrapQ1(128, 0, 1)))
print("Part A, n=256: " + str(compTrapQ1(256, 0, 1)))
print("Part A, n=512: " + str(compTrapQ1(512, 0, 1)))

print("Part B, n=128: " + str(compTrapQ2(128, 0, 2 * math.pi)))
print("Part B, n=256: " + str(compTrapQ2(256, 0, 2 * math.pi)))
print("Part B, n=512: " + str(compTrapQ2(512, 0, 2 * math.pi)))
