import numpy as np
import matplotlib.pyplot as plt

class PolynomialDerivative:
    def __init__(self, n):
        """
        Initializes the polynomial with a given highest degree 'n'.
        The polynomial is represented as x^n + x^(n-1) + ... + x^1 + x^0.
        
        :param n: The degree of the polynomial.
        """
        self.n = n  # Highest degree of the polynomial
    
    def f_s(self, x, order=0):
        """
        This method returns the coefficients of the polynomial or its derivatives.
        
        :param x: The value at which the polynomial (or its derivatives) is evaluated.
        :param order: The order of the derivative. 0 for the polynomial itself, 1 for the first derivative,
                      2 for the second derivative.
        :return: A list of coefficients for the polynomial or its derivatives at x.
        """
        # Generate the polynomial coefficients (from x^n to x^0)
        coefficients = [x**i for i in range(self.n, -1, -1)]
        
        # For order 0, return the polynomial itself
        if order == 0:
            return np.array(coefficients).reshape(1, -1)
        
        # For order 1 (first derivative), calculate the first derivative coefficients
        elif order == 1:
            coefficients_deriv = []
            for i in range(self.n, 0, -1):
                coefficients_deriv.append(i * x**(i-1))
            coefficients_deriv.append(0)  # The derivative of the constant term is 0
            return np.array(coefficients_deriv).reshape(1, -1)
        
        # For order 2 (second derivative), calculate the second derivative coefficients
        elif order == 2:
            coefficients_deriv_2 = []
            for i in range(self.n, 1, -1):
                coefficients_deriv_2.append(i * (i-1) * x**(i-2))
            coefficients_deriv_2.extend([0, 0])  # The second derivative of x^1 and constant term is 0
            return np.array(coefficients_deriv_2).reshape(1, -1)
        
        # Order 3: Third derivative
        elif order == 3:
            coefficients_deriv_3 = []
            for i in range(self.n, 2, -1):
                coefficients_deriv_3.append(i * (i-1) * (i-2) * x**(i-3))
            coefficients_deriv_3.extend([0, 0, 0])  # The third derivative of x^2, x^1, and constant term is 0
            return np.array(coefficients_deriv_3).reshape(1, -1)

        
        # If the order is not 0, 1, or 2, raise an exception
        else:
            raise ValueError("Only order 0, 1, or 2 are supported for derivatives.")

    def evaluate_polynomial(self, x, render_s):
        """
        Renders a polynomial based on the provided coefficients and rendering points, with the highest degree of self.n.
        
        :param x: A list or array of coefficients (should be of length self.n + 1).
        :param render_s: A set of values at which the polynomial will be evaluated (numpy array).
        :return: render_x and render_y: The evaluated polynomial values at render_s.
        """
        if len(x) != (self.n + 1)*2:
            raise ValueError(f"Expected {self.n + 1} coefficients, but got {len(x)}.")

        xs, ys = x[:self.n+1], x[self.n+1:]
        render_x, render_y = [], []

        for _, val in enumerate(render_s):
            tmp_x = 0
            tmp_y = 0
            for i in range(self.n+1):
                tmp_x += xs[i] * val**(self.n-i)
                tmp_y += ys[i] * val**(self.n-i)
            render_x.append(tmp_x)
            render_y.append(tmp_y)
        
        return np.ravel(np.column_stack((render_x, render_y)))
    
    def evaluate_velocity(self, x, render_t, order=0):
        """
        Renders a polynomial based on the provided coefficients and rendering points, with the highest degree of self.n.
        
        :param x: A list or array of coefficients (should be of length self.n + 1).
        :param render_t: A set of values at which the polynomial will be evaluated (numpy array).
        :return: render_y: The evaluated polynomial values at render_t.
        """
        if len(x) != (self.n + 1):
            raise ValueError(f"Expected {self.n + 1} coefficients, but got {len(x)}.")

        render_y = []
        for _, val in enumerate(render_t):
            array_dot = self.f_s(val, order) @ np.array(x)
            render_y.append(array_dot[0])
        
        return np.array(render_y)
    

