import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x_val = np.linspace(0, 6, 100, dtype = float)

# bias
'''
bias: the difference between the predicted values of a model and the correct value in training set

i.e. high bias models are inflexible models (think linear regression)
'''
bias_y_val = (1 / x_val ** 3) + 1


# variance 
'''
variance: the variability of a model prediction for a given data point

i.e. high variability models are extremely flexible models (think polynomial regression)
'''


# test error
''' test error:'''
y_val_test_error = ((x_val -3) ** 2) + 2

plt.axis(ymax = 10, ymin = 0, xmax = 6, xmin = 0)
plt.plot(x_val, y_val_test_error, "--", label = "test error")
plt.plot(x_val, bias_y_val, "-", label = "bias squared")
plt.legend()
plt.show()
