import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

complexity = np.linspace(0, 6, 100, dtype = float)

# bias
'''
bias: the difference between the predicted values of a model and the correct value in training set

i.e. high bias models are inflexible models (think linear regression)
'''
bias_y_val = (1/np.exp(complexity)) * 11
              

# variance 
'''
variance: the variability of a model prediction for a given data point

i.e. high variability models are more flexible models (think polynomial regression)
'''
variance_y_val = np.exp(complexity) / 36

# test error
''' test error: the difference between the true value in the test set and the value that 
we predicted in the test set.'''
y_val_test_error = ((complexity -3) ** 2) + 2

plt.axis(ymax = 10, ymin = 0, xmax = 6, xmin = 0)
plt.plot(complexity, y_val_test_error, "--", label = "test error")
plt.plot(complexity, bias_y_val, ".", label = "bias squared")
plt.plot(complexity, variance_y_val, ".", label = "variance")
plt.legend()
plt.xlabel("Complexity")
plt.ylabel("Error")
plt.show()


'''
the main lessons here are that 
1) bias means low flexibilty model
2) variance means high flexibility model
3) bias variance tradeoff is that as we more from more biased to more flexible models,
we get decreasing, then increasing test error. i.e. our predictions get better to a point,
but then after that point our test predictions get worse.
'''