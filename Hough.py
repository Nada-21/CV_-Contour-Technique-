import numpy as np
import cv2
import math
from collections import defaultdict

#.......................................Line Detection.....................................................

def line_detection(image, edge_image, threshold):
  
    edge_height, edge_width = edge_image.shape[:2]
    
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))   #Diagonal 

    thetas = np.arange(0, 180, 1)    #theta: The resolution of the parameter θ in radians.
    rhos = np.arange(-d, d, 1)       #rho : The resolution of the parameter r in pixels.
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    
    accumulator = np.zeros((len(rhos), len(thetas)))      #Hough Accumulator

    y_idxs, x_idxs = np.nonzero(edge_image)    # find all edge (nonzero) pixel indexes
  
    for i in range(len(x_idxs)):               # cycle through edge points
        x = x_idxs[i]
        y = y_idxs[i]

        for theta_idx in range(len(thetas)):           # cycle through thetas and calc rho
            rho = int((x * cos_thetas[theta_idx] +
                y * sin_thetas[theta_idx]) + d )
            
            accumulator[rho, theta_idx] += 1

    for y in range(accumulator.shape[0]):
        for x in range(accumulator.shape[1]):
            if accumulator[y][x] > threshold:
                rho = rhos[y]
                theta = thetas[x]
                a = np.cos(np.deg2rad(theta))
                b = np.sin(np.deg2rad(theta))
                x0 = (a * rho)      # x0 stores the value rcos(theta)
                y0 = (b * rho)      # y0 stores the value rsin(theta)

                #these are then scaled so that the lines go off the edges of the image
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                out_img = cv2.line(image, (x1,y1), (x2,y2), (255, 0, 0), 2)

    return out_img

#.................................Circle Detection................................................................

def circle_detection(input_img,edge_image,r_min,r_max,delta_r):
    num_thetas = 100
    bin_threshold = 0.4
    post_process = True
    img_height, img_width = edge_image.shape[:2]
    dtheta = int(360 / num_thetas)
    thetas = np.arange(0, 360, step=dtheta)
    rs = np.arange(r_min, r_max, step=delta_r)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    circle_candidates = []

    for r in rs:
        for t in range(num_thetas):
          circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
    accumulator = defaultdict(int)

    for y in range(img_height):
        for x in range(img_width):
            if edge_image[y][x] != 0:
                for r, rcos_t, rsin_t in circle_candidates:
                    x_center = x - rcos_t
                    y_center = y - rsin_t
                    accumulator[(x_center, y_center, r)] += 1 #vote for current candidate
    output_img = input_img.copy()
    out_circles = []

    for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
     x, y, r = candidate_circle
     current_vote_percentage = votes / num_thetas
     if current_vote_percentage >= bin_threshold:
          out_circles.append((x, y, r, current_vote_percentage))
    if post_process :
        pixel_threshold = 5
        postprocess_circles = []
        for x, y, r, v in out_circles:
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
                postprocess_circles.append((x, y, r, v))
        out_circles = postprocess_circles

    for x, y, r, v in out_circles:

        output_img = cv2.circle(output_img, (x,y), r, (0,255,0), 2)
    return output_img

#.......................................Ellipse Detection.....................................................
def Ellipse_detection(image_rgb,edge_image, min2a = 10, min_votes = 10):
     w, h = edge_image.shape
     EdgePixel = []
     output_img = image_rgb.copy()
        # Step 1 - Save all Edge-Pixels in an array
     for x in range(w):
            for y in range(h):
                if edge_image[x, y] != 0:
                 EdgePixel.append([x,y])
     acc = np.zeros(int(max(w, h)/2))
     for i in range(len(EdgePixel)-1):
         for j in range(i+2,len(EdgePixel)):
             x1= EdgePixel[i][0]
             y1 = EdgePixel[i][1]
             x2=EdgePixel[j][0]
             y2=EdgePixel[j][1]
             d12=np.linalg.norm(np.array([x1,y1])-np.array([x2,y2]))
             acc = acc * 0
             if  x1 - x2 > min2a and d12 > min2a :
                #  center
                x0 = (x1 + x2)/2
                y0 = (y1 + y2)/2
                # Half-length of the major axis.
                a = d12/2
                # Orientation.
                if x2!=x1:
                 alpha = math.atan((y2 - y1)/(x2 - x1))
                else:
                    alpha=0
                # Distances between the two points and the center.
                d01 = np.linalg.norm(np.array([x1, y1]) - np.array([x0, y0]))
                d02 = np.linalg.norm(np.array([x2, y2]) - np.array([x0, y0]))
                for k in range(len(EdgePixel)):
                    if k == i and k == j:
                        continue

                    x3= EdgePixel[k][0]
                    y3 = EdgePixel[k][1]
                    d03 = np.linalg.norm(np.array([x3, y3]) - np.array([x0, y0]))
                    if  d03 >= a:
                        continue
                    # Distance f
                    f= np.linalg.norm(np.array([x3, y3]) - np.array([x2, y2]))
                    # Estimating the half-length of the minor axis.
                    cos2_tau = pow(((pow(a, 2)+ pow(d03, 2) - pow(f, 2)) / (2 * a * d03)), 2)
                    sin2_tau = 1 - cos2_tau
                    b = math.sqrt(abs((pow(a, 2) * pow(d03, 2) * sin2_tau) /(pow(a, 2) - pow(d03, 2) * cos2_tau)))
                    # Changing the score of the accumulator, if b is a valid value.
                    #  NOTE: the accumulator's length gives us the biggest expected
                    #  value for b, which means, in this current implementation,
                    #  we wouldn't detect ellipses whose half of minor axis is
                    #  greater than the image's size (look at the acc's declaration).
                    if b > 0 and b <= len(acc):
                        acc[int(b)]= acc[int(b)]+1

                sv=np.argmax(acc)
                si=np.max(acc)
                if sv > min_votes:
                    #  Ellipse detected!
                    #  The index si gives us the best b value.
                     parameters = [x0, y0, a, si, alpha]
                    #  return parameters
                     center_coordinates=(int(x0),int(y0))
                     axesLength=(int(2*a),int(2*si))
                     output_img=cv2.ellipse(output_img, center_coordinates, axesLength,alpha,(0,255,0), 2)
                     return output_img
                else:
                    return output_img







