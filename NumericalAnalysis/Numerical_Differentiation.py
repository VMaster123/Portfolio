import matplotlib.pyplot as plt


def functions(x):
    return x**2


def centralDifferentiation(delta_t, func, minRange, maxRange):
    points = []
    t = minRange
    while t <= maxRange:
        y_value = (func(t + delta_t) - func(t)) / (2 * delta_t)
        points.append((t, y_value))
        t += delta_t
    return points


points = centralDifferentiation(0.01, functions, 0, 100)


x_val, y_val = zip(*points)

plt.plot(x_val, y_val, label="d/dx")
plt.show()
