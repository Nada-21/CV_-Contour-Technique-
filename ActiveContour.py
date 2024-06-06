from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pylab as plb
import os
import copy
import matplotlib.cm as cm
from PIL import Image
from random import randint



#---------------------------------------------------Snake Contour-------------------------------------------------------

def create_A(a, b, N):
    """
    a:alpha parameter
    b:beta parameter
    N:is the number of points sampled on the snake curve: (x(p_i), y(p_i)), i=0,...,N-1
    """
    row = np.r_[
        -2*a - 6*b, 
        a + 4*b,
        -b,
        np.zeros(N-5),
        -b,
        a + 4*b
    ]
    A = np.zeros((N,N))
    for i in range(N):
        A[i] = np.roll(row, i)
    return A


def create_external_edge_force_gradients_from_img( img, sigma=30. ):
    """
    Given an image, returns 2 functions, fx & fy, that compute
    the gradient of the external edge force in the x and y directions.
    img: ndarray
        The image.
    """
    img=(img-img.min()) / (img.max()-img.min())
    # Gaussian smoothing.
    smoothed = cv2.GaussianBlur(img, (7,7), 30)
    # Gradient of the image in x and y directions.
    gix=cv2.Sobel(smoothed ,cv2.CV_64F, 1, 0, ksize=3)
    giy =cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    # Gradient magnitude of the image.
    gmi = (gix**2 + giy**2)**(0.5)
    # Normalize. This is crucial (empirical observation).
    gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())
     # Gradient of gradient magnitude of the image in x and y directions.
    ggmiy, ggmix = np.gradient( gmi )
    def fx(x, y):
        """
        Return external edge force in the x direction.
        """
        # Check bounds.
        x[ x < 0 ] = 0.
        y[ y < 0 ] = 0.

        x[ x > img.shape[1]-1 ] = img.shape[1]-1
        y[ y > img.shape[0]-1 ] = img.shape[0]-1

        return ggmix[ (y.round().astype(int), x.round().astype(int)) ]
    def fy(x, y):
        """
        Return external edge force in the y direction.
        """
        # Check bounds.
        x[ x < 0 ] = 0.
        y[ y < 0 ] = 0.

        x[ x > img.shape[1]-1 ] = img.shape[1]-1
        y[ y > img.shape[0]-1 ] = img.shape[0]-1

        return ggmiy[ (y.round().astype(int), x.round().astype(int)) ]

    return fx, fy


def iterate_snake(img,x, y, a, b, gamma=0.1, n_iters=10, return_all=True):
    """
    x:intial x coordinates of the snake
    y:initial y coordinates of the snake
    a:alpha parameter
    b:beta parameter
    fx: callable
        partial derivative of first coordinate of external energy function. This is the first element of the gradient of the external energy.
    fy: callable
        see fx.
    gamma:step size of the iteration
    
    n_iters:number of times to iterate the snake
    return_all: if True, a list of (x,y) coords are returned corresponding to each iteration.
        if False, the (x,y) coords of the last iteration are returned.
    """
    # fx and fy are callable functions
    fx, fy = create_external_edge_force_gradients_from_img( img, sigma=10 )
    
    A = create_A(a,b,x.shape[0])
    B = np.linalg.inv(np.eye(x.shape[0]) - gamma*A)
    if return_all:
        snakes = []

    for i in range(n_iters):
        x_ = np.dot(B, x + gamma*fx(x,y))
        y_ = np.dot(B, y + gamma*fy(x,y))
        x, y = x_.copy(), y_.copy()
        if return_all:
            snakes.append( (x_.copy(),y_.copy()) )

    if return_all:
        return snakes
    else:
        return (x,y)
    
def initialcontour(img,R):        #R:radius
    cx=img.shape[0]/2            #cx,cy:center
    cy=img.shape[1]/2
    t = np.arange(0, 2*np.pi, 0.1)
    x = cx+R*np.cos(t)
    y = cy+R*np.sin(t)    
    return x ,y 


#---------------------------------------------------Chain Code---------------------------------------------------------
'''using them to call  passes_through_all_points function
x_axis=np.r_[snakes[-1][0], snakes[-1][0][0]]
y_axis=(np.r_[snakes[-1][1], snakes[-1][1][0]])
arr = np.stack((x_axis, y_axis), axis=1)'''

# Python3 code for generating 8-neighbourhood chain

codeList = [5, 6, 7, 4, -1, 0, 3, 2, 1]
# This function generates the chaincode
# for transition between two neighbour points
def getChainCode(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    hashKey = 3 * dy + dx + 4
    return codeList[hashKey]

'''This function generates the list ofchaincodes for given list of points'''
def generateChainCode(ListOfPoints):
    chainCode = []
    for i in range(len(ListOfPoints) - 1):
        a = ListOfPoints[i]
        b = ListOfPoints[i + 1]
        chainCode.append(getChainCode(a[0], a[1], b[0], b[1]))
    return chainCode

'''This function generates the list of points for
a straight line using Bresenham's Algorithm'''
def Bresenham2D(x1, y1, x2, y2):
    ListOfPoints = []
    ListOfPoints.append([x1, y1])
    xdif = x2 - x1
    ydif = y2 - y1
    dx = abs(xdif)
    dy = abs(ydif)
    if(xdif > 0):
        xs = 1
    else:
        xs = -1
    if (ydif > 0):
        ys = 1
    else:
        ys = -1
    if (dx > dy):
 
        # Driving axis is the X-axis
        p = 2 * dy - dx
        while (x1 != x2):
            x1 += xs
            if (p >= 0):
                y1 += ys
                p -= 2 * dx
            p += 2 * dy
            ListOfPoints.append([x1, y1])
    else:
 
        # Driving axis is the Y-axis
        p = 2 * dx-dy
        while(y1 != y2):
            y1 += ys
            if (p >= 0):
                x1 += xs
                p -= 2 * dy
            p += 2 * dx
            ListOfPoints.append([x1, y1])
    return ListOfPoints

def DriverFunction(x1,y1,x2,y2):
    (x1, y1) = (x1,y1 )
    (x2, y2) = (x2, y2)
    ListOfPoints = Bresenham2D(x1, y1, x2, y2)
    chainCode = generateChainCode(ListOfPoints)
    chainCodeString = "".join(str(e) for e in chainCode)
    print ('Chain code for the straight line from', (x1, y1),'to', (x2, y2), 'is', chainCodeString)
    print (chainCode)
    print (ListOfPoints)
    
    
def passes_through_all_points (arr):
    for i in range(len(arr)-1):
        x1=int(arr[i][0])
        y1=int(arr[i][1])
        x2=int(arr[i+1][0])
        y2=int(arr[i+1][1])
        DriverFunction(x1,y1,x2,y2) 
     
    
#---------------------------compute_perimeter & compute_area--------------------------
def compute_perimeter(contour):
    """
    This function takes a contour as input and computes its perimeter
    """

    # Start with the first point of the contour
    current_point = contour[0]
    perimeter = 0

    # Iterate over all points of the contour
    for i in range(1, len(contour)):
        # Find the next point on the contour
        next_point = contour[i]
        # Compute the euclidean distance between the points
        distance = math.sqrt((next_point[0] - current_point[0])**2 + (next_point[1] - current_point[1])**2)
        # Add the distance to the perimeter
        perimeter += distance
        # Move to the next point
        current_point = next_point
        
    return perimeter

def compute_area(contour):
    """
    This function takes a contour as input and computes the area inside it
    """

    # Convert the contour to a list of x-coordinates and y-coordinates
    x_values = [point[0] for point in contour]
    y_values = [point[1] for point in contour]

    # Apply the shoelace formula to compute the area
    area = 0.5 * abs(sum(x_values[i] * y_values[(i + 1) % len(contour)] - x_values[(i + 1) % len(contour)] * y_values[i]
                         for i in range(len(contour))))
    return area

        
        
