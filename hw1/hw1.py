import numpy as np
import matplotlib.pyplot as plt

def Euler(derivative, x, y, interval=0.01):
    
    for n in range(0, len(x) - 1):    # 1-1000
        
        # y_n+1 = y_n + f'(x) * (x_n+1 - x_n)
        y[n + 1] = y[n] + derivative(x[n], y[n]) * interval
    
def Modified_Euler(derivative, x, y, interval=0.01):
    
    for n in range(0, len(x) - 1):    # 1-1000
        
        # 1st estimation of y_n+1
        y[n + 1] = y[n] + derivative(x[n], y[n]) * interval
        y[n + 1] = y[n] + (derivative(x[n], y[n]) + derivative(x[n + 1], y[n + 1])) * interval / 2

def RK4(derivative, x, y, interval=0.01):
    
    for n in range(0, len(x) - 1):
        
        # estimated slope
        k1 = derivative(x[n], y[n])
        k2 = derivative(x[n] + interval / 2, y[n] + k1 * interval / 2)
        k3 = derivative(x[n] + interval / 2, y[n] + k2 * interval / 2)
        k4 = derivative(x[n] + interval, y[n] + interval * k3)
        
        # final estimation
        y[n + 1] = y[n] + (k1 + 2 * k2 + 2 * k3 + k4) * interval / 6
    
def derivative(x, y):
    return 5 * np.cos(-1 * np.abs(x * y) / 5)


if __name__ == "__main__":
    
    # generate x index
    x = np.linspace(0, 10, 1001)
    y = np.zeros_like(x)
    
    fig, ax = plt.subplots()
    # Euler plot
    Euler(derivative, x, y)
    ax.plot(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Euler method")
    ax.grid(visible=True)
    fig.savefig("Euler")
    fig.show()
    fig.waitforbuttonpress()
    
    fig, ax = plt.subplots()
    y = np.zeros_like(x)
    # Modified-Euler plot
    Modified_Euler(derivative, x, y)
    ax.plot(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Modified-Euler method")
    ax.grid(visible=True)
    fig.savefig("Modified-Euler")
    fig.show()
    fig.waitforbuttonpress()
    
    fig, ax = plt.subplots()
    y = np.zeros_like(x)
    # RK4 plot
    RK4(derivative, x, y)
    ax.plot(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("RK4 method")
    ax.grid(visible=True)
    fig.savefig("RK4")
    fig.show()
    fig.waitforbuttonpress()
    