import matplotlib.pyplot as plt
import numpy as np


# error as in int of signifgant digits
def bisection(error, a, b, midpoint, func, limit):
    while limit < 50:
        if int(func(midpoint) * error) == 0:
            return midpoint
        else:
            if func(a) * func(midpoint) > 0:
                a = midpoint
                midpoint = (a + b) / 2
                bisection(error, a, b, midpoint, func, limit + 1)
            else:
                b = midpoint
                midpoint = (a + b) / 2
                bisection(error, a, b, midpoint, func, limit + 1)


x_vals = np.linspace(0, 2 * np.pi, 100)
y_vals = np.cos(x_vals)
midpoint = bisection(200, np.pi / 2, np.pi, (np.pi / 2 + np.pi) / 2, np.cos, 0)

print(midpoint)
plt.plot(x_vals, y_vals, label="function")

plt.scatter(midpoint, 0)
plt.show()
