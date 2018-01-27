from numpy import *

def compute_error(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b, m, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m * x) + b))
        m_gradient += -(2/N) * x * (y - ((m * x) + b))
    new_b = b - (learningRate * b_gradient)
    new_m = m - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent(points, init_b, init_m, learning_rate, iterations):
    b = init_b
    m = init_m
    for i in range(iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    iterations = 10000
    print("Gradient descent at b = {0}, m = {1}, error = {2}".format(
        initial_b, initial_m, compute_error(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent(points, initial_b, initial_m, learning_rate, iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(
        iterations, b, m, compute_error(b, m, points)))

if __name__ == '__main__':
    run()
