import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



class regression_line:

    #find correlation coefficent
    def correlation_coef(self, data):

        n = len(data[0])
        x_sum = np.sum(data[0])
        y_sum= np.sum(data[1])
        xy_sum = np.sum(np.multiply(data[0], data[1]))



        SPxy = (xy_sum - ((x_sum * y_sum) / n))

        SPx = np.sum(data[0] ** 2) - ((x_sum ** 2) / n)

        SPy = np.sum(data[1] ** 2) - ((y_sum ** 2) / n)

        return(SPxy / math.sqrt(SPx * SPy))

    #find slope 
    def slope_fun(self, data, cor):
        return(cor * (math.sqrt((np.sum((data[1] - np.mean(data[1])) ** 2)) / (np.sum((data[0] - np.mean(data[0])) ** 2)))))

    #find intercept
    def intercept_fun(self, data, b):
        return(np.mean(data[1]) - (b * np.mean(data[0])))

    def __init__(self, data):
        self.correlation = self.correlation_coef(data)
        self.slope = self.slope_fun(data, self.correlation)
        self.intercept = self.intercept_fun(data, self.slope)
        self.n = len(data[0])
        self.copy_of_data = data


    def predict(self, x):
        return (self.slope * x + self.intercept)



    def standard_error_of_estimate(self):

        print(math.sqrt((np.sum((self.copy_of_data[1] - np.mean(self.copy_of_data[1]) ** 2)))
        (1 - (self.correlation ** 2)) / (self.n - 2)))




data = np.array([[5, 5, 2, 2, 3, 1, 2], [4, 3, 2, 2, 2, 1, 2]])


new_line = regression_line(data)



predicted_y = new_line.predict(data[0])


plt.scatter(data[0], data[1])
plt.plot(data[0], predicted_y)

plt.show()