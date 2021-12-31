# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np

from matplotlib.colors import ListedColormap

# Global Variables
A = 0.17
XMIN = -3
XMAX = 3
YMIN = -2
YMAX = 2

XNUM = 1000

MAXITERATIONS = 100
colorMap = "magma"
cycleSize = 200

juliaConstant = -0.76 + 0.08j


def Colors(I):
    colors = I % cycleSize
    colors[I == MAXITERATIONS - 1] = cycleSize * 1.1
    return colors


def Mandlebrot(xCoords, yCoords, xNum, maxIterations, squareComp=0):
    (xMin, xMax) = xCoords
    (yMin, yMax) = yCoords

    yNum = int(xNum * (yMax - yMin) / (xMax - xMin))
    xVals = np.linspace(xMin, xMax, xNum).reshape(1, xNum)
    yVals = np.linspace(yMin, yMax, yNum).reshape(yNum, 1)

    C = np.tile(xVals, (yNum, 1)) + 1j * np.tile(yVals, (1, xNum))

    Z = np.zeros((yNum, xNum), dtype=complex)
    M = np.full((yNum, xNum), True, dtype=bool)
    I = np.zeros((yNum, xNum), dtype=int)

    if squareComp == 0:
        for i in range(maxIterations):
            Z[M] = Z[M] * Z[M] + C[M]
            I[np.abs(Z) < 2] = i
            M[np.abs(Z) > 2] = False

    else:
        for i in range(maxIterations):
            Z[M] = Z[M] * Z[M] + C[M] + (C[M] * C[M]) * squareComp
            I[np.abs(Z) < 2] = i
            M[np.abs(Z) > 2] = False

    return I


def Julia(xCoords, yCoords, xNum, maxIterations):
    (xMin, xMax) = xCoords
    (yMin, yMax) = yCoords

    yNum = int(xNum * (yMax - yMin) / (xMax - xMin))

    xVals = np.linspace(xMin, xMax, xNum).reshape(1, xNum)
    yVals = np.linspace(yMin, yMax, yNum).reshape(yNum, 1)

    Z = np.tile(xVals, (yNum, 1)) + 1j * np.tile(yVals, (1, xNum))

    C = juliaConstant
    M = np.full((yNum, xNum), True, dtype=bool)
    I = np.zeros((yNum, xNum), dtype=int)

    for i in range(maxIterations):
        Z[M] = Z[M] * Z[M] + C
        I[np.abs(Z) < 2] = i
        M[np.abs(Z) > 2] = False

    return I


def animate(i):
    a = 0.10 + i * 0.0002
    ax.clear()  # clear axes object
    ax.set_xticks([], [])  # clear x-axis ticks
    ax.set_yticks([], [])  # clear y-axis ticks

    xCoords = (XMIN - 0.5 / a, XMAX - 0.5 / a)
    yCoords = (YMIN, YMAX)

    I = Mandlebrot(xCoords, yCoords, XNUM, MAXITERATIONS, a)

    colormap = cm.get_cmap(colorMap, int(cycleSize*1.1))
    newcolors = colormap(np.linspace(0, 1, 256))
    black = np.array([0,0,0,1])
    newcolors[-1, :] = black
    newcmp = ListedColormap(newcolors)

    colors = Colors(I)
    # associate colors to the iterations with an interpolation
    img = ax.imshow(colors, interpolation="bicubic", cmap=newcmp)
    print(f"Frame {i} completed")
    return [img]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DPI = 1920 / 15
    fig = plt.figure(figsize=(15, 10))  # instantiate a figure to draw
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = plt.axes()  # create an axes object
    plt.axis("off")
    anim = animation.FuncAnimation(fig, animate, frames=600, interval=33, blit=True)
    anim.save('mandelbrot.gif', writer='imagemagick', dpi=DPI)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
