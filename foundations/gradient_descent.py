class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x=init
        func=x**2
        der=2*x

        x_new=x
        for step in range(iterations):
            if step==0:
                x_old=x
                x_new=x_old - learning_rate*der
                x_old=x_new
                continue;
            der_new=2*x_old
            x_new=x_old - learning_rate*der_new
            x_old=x_new
           
        return round(x_new,5)   
