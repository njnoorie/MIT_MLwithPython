


import numpy as np

def perceptron(feature_matrix, labels, T):
    # Ensure feature_matrix and labels are numpy arrays
    feature_matrix = np.array(feature_matrix, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)

    # Initialize theta and theta_0
    current_theta = np.zeros(feature_matrix.shape[1], dtype=np.float64)
    current_theta_0 = np.float64(0)

    for t in range(T):  # Loop through dataset T times
        for i in range(feature_matrix.shape[0]):  # Iterate through each sample
            fx = labels[i] * (np.dot(feature_matrix[i], current_theta) + current_theta_0)
            if fx <= 0:
                current_theta += feature_matrix[i] * labels[i]
                current_theta_0 += labels[i]

    # Convert theta to a list of strings with 7 decimal places
    #theta_output = [f"{x:.7f}" for x in current_theta]
    return current_theta, current_theta_0



feature_matrix = np.array([
    [0.40431434, 0.17163667, 0.31416353, 0.33193044, 0.07487755, -0.15652913, -0.49976739, -0.27968083, 0.31628542, 0.00288022],
    [-0.35107078, 0.27509508, 0.10751677, 0.18890273, 0.18913872, -0.314942, -0.01404542, 0.25051729, -0.07336076, -0.42071251],
    [0.29799737, 0.48361542, -0.14364622, -0.26230225, 0.2844714, -0.0911066, -0.18222994, -0.44567855, -0.41385513, 0.34409836],
    [-0.22608604, -0.00889771, -0.13044362, 0.23124749, 0.37463665, -0.37600381, 0.2138942, -0.05755614, -0.41637598, 0.01534453],
    [-0.48778402, 0.08885646, 0.06926756, -0.36836009, -0.13036904, -0.069639, -0.32658948, 0.08895031, -0.11395868, 0.490259]
])
labels = np.array([-1, 1, -1, 1, 1])
T = 5
expected_theta=['-1.3098648', '-0.8872721', '0.2261164', '0.3874919', '-0.3246752', '-0.2634296', '0.2517646', '0.9227513', '0.2973756', '-0.1825932']

output = perceptron(feature_matrix, labels, T)
print(output)