import matplotlib.pyplot as plt
import numpy as np

points_2d = [(-2, 3), (-1, 7), (0, 1), (1, 0), (2, 2), (3, 3)]
current_stage_points = []
interp_points = []


def dividedDifference(current_stage_points, points_2d, interp_points):
    current_stage = (
        0
        if (len(current_stage_points) == 0)
        else len(points_2d) - len(current_stage_points)
    )
    if current_stage == len(points_2d) - 1:
        interp_points.append(current_stage_points[0])
        return interp_points
    else:
        if current_stage == 0:
            interp_points.append(points_2d[0][1])
            for i in range(len(points_2d) - 1):
                current_stage_points.append(
                    (points_2d[i + 1][1] - points_2d[i][1])
                    / (points_2d[i + 1][0] - points_2d[i][0])
                )
        else:
            interp_points.append((current_stage_points[0]))

            for i in range(len(current_stage_points) - 1):
                current_stage_points[i] = (
                    current_stage_points[i + 1] - current_stage_points[i]
                ) / (points_2d[i + current_stage + 1][0] - points_2d[i][0])
            del current_stage_points[len(current_stage_points) - 1]
        return dividedDifference(current_stage_points, points_2d, interp_points)


lister = dividedDifference(current_stage_points, points_2d, interp_points)
print("--------------------------------------")


def interpolation(list, points_2d):
    finalPoly = np.poly1d([0])
    for i in range(len(list)):
        roots = [points_2d[j][0] for j in range(i)]
        polynomial = np.poly1d(np.poly(roots))
        polynomial *= list[i]
        finalPoly += polynomial
    return finalPoly


print("-----------------------------Question 4----------------------------------")

finalPoly = interpolation(lister, points_2d)
print("Polynomial for Question 4:")

# question 4 polynomial interpolated
print(finalPoly)
"""
This part prints the polynomial out and the points I used to make the polynomial
x_vals = np.linspace(-2, 3, 400)
y_vals = finalPoly(x_vals)

plt.plot(x_vals, y_vals, color="r")

original_x = [c[0] for c in points_2d]
original_y = [a[1] for a in points_2d]
plt.scatter(original_x, original_y, color="g")
plt.grid(True)
"""

print("-----------------------------Question 5----------------------------------")


def polynomomial(x):
    return 2 / (1 + x**2)


x_org = np.linspace(-5, 5, 400)
y_org = polynomomial(x_org)

current_stage_points = []
interp_points = []

pointsp2 = [(-5, 1 / 13), (0, 2), (5, 1 / 13)]
pointsp4 = [
    (-5, 1 / 13),
    (-2.5, 2 / (1 + 2.5**2)),
    (0, 2),
    (2.5, 2 / (1 + 2.5**2)),
    (5, 1 / 13),
]

coeffsp2 = dividedDifference(current_stage_points, pointsp2, interp_points)
finalPolynomial2 = interpolation(coeffsp2, pointsp2)
print("p2 polynomial")
# p2 polynomial
print(finalPolynomial2)

current_stage_pointsp4 = []
interp_pointsp4 = []

coeffsp4 = dividedDifference(current_stage_pointsp4, pointsp4, interp_pointsp4)
finalPolynomial4 = interpolation(coeffsp4, pointsp4)
print("p4 polynomial")
# p4 polynomial
print(finalPolynomial4)

x_vals = np.linspace(-5, 5, 400)
y_valsp2 = finalPolynomial2(x_vals)
y_valsp4 = finalPolynomial4(x_vals)

# This printed out the original graph and the p_2(x) and p_4(x) interpolation polynomials
# plt.plot(x_vals, y_org, label="original graph", color="y")
# plt.plot(x_vals, y_valsp2, label="p2 interpolation", color="r")
# plt.plot(x_vals, y_valsp4, label="p4 interpolation", color="g")

original_x = [c[0] for c in pointsp2]
original_y = [a[1] for a in pointsp2]

# p2 points I used to interpolate
# plt.scatter(original_x, original_y, color="g")
original_x_4 = [t[0] for t in pointsp4]
original_y_4 = [z[1] for z in pointsp4]

# p4 points I used to interpolate
# plt.scatter(original_x_4, original_y_4, color="y")

# p2 error graph
# plt.plot(x_vals, y_valsp2 - y_org, label="p2 error", color="r")

# p4 error graph
plt.plot(x_vals, y_valsp4 - y_org, label="p4 error", color="g")
plt.grid(True)
plt.legend()
plt.show()
