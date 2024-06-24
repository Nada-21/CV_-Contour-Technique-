# CV-A02 : Contour Technique

## Description:
A small web application based app developed with python and streamlit, to apply different image processing techniques.

## Requirements:
•	Python 3.  
•	Streamlit 1.13.0  
•	Numpy 1.23.4  
•	Matplotlib 3.6.2  

## Running command:
Streamlit run server.py   

-The UI contains two main tabs Hough Transformations, Active Contour  

# Tab1:
•	Line Detection  
•	Circle Detection     
•	Ellipse Detection   

### Line Detection:
#### Algorithm
First: We read image using cv2 library then, apply canny edge detection also using cv2 library.   
Second: Call line detection function with parameters uploaded image, edged image and threshold from user, we want to determine the height & width of edged image to determine the diagonal then, determine “theta” resolution and “r” resolution.  
Then, find all non-zeros edges pixels and cycle through them and cycle through theta to calculate “r” resolution to determine hough accumulator.  
Last step:  Cycle through accumulator to determine line points then, using cv2.line library to superimpose the detected line on the input image.  
#### Parameters that user can enter them:
•	Line Detection Threshold   
•	Canny Edge Lower Threshold  
•	Canny Edge Higher Threshold    
![Screenshot (1470)](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/08d6b736-23a5-434f-a02b-dc641b8c9b50)  
Result  
![Screenshot (1471)](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/4f97d563-f364-4155-80d1-a08732afba0a)  

### Circle Detection:
#### Algorithm
First: We read image using cv2 library then, apply canny edge detection also using cv2 library.   
Second: Call circle detection function with parameters uploaded image, edged image, min radius and max radius of detected circle and delta r change from min radius and max radius. Based on defined number of angles we calculate angles from 0 to 360 to build parametric equation of circle (x = x_center + r * cos(t) and y = y_center + r * sin(t)), then append it in array of candidate circles. loop on image height and width of an image to find an edge pixel which pass through any of candidate circle array then build accumulator from defined current circles in image.  
Last step:  Sort accumulator in addition to some post process to exclude too close circles and duplicated ones. Finally using cv2.cirlce library to superimpose the detected circles on the input image.
#### Parameters that user can choose them:
•	Canny Edge Lower Threshold  
•	Canny Edge Higher Threshold  
•	Minimum Circle Radius   
•	Maximum Circle Radius   
•	Change between min radius and max radius.  
![Screenshot (1472)](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/cea9d7c2-cda1-454e-a2f0-c8492d54a25c)  
Result  
![Screenshot (1473)](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/9e8385f0-13b2-4a4d-b9ba-c4e3640dc7d7)  
# Tab2:
### Active Contour (Snake)
#### Algorithm
Snake theory:  
![image](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/8a6047c1-bb0b-4392-ac09-878a1568e0cb)  
![image](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/d285826f-ad30-456a-9822-a6604ca4d711)
![image](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/c1e27c9b-7766-4b56-9ac5-1d1b2a290feb)
![image](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/f27f8c43-c5e5-43f1-89db-796d3c2090e1)
![image](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/5ae75772-3970-476e-8e27-0116669ec211)
#### Parameters that the user can enter them:
•	Radius  
•	Alpha  
•	Beta  
•	Gamma  
•	Num_of_iterations 
Result:
Initial contour
![Screenshot (1474)](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/d96ffe09-ebc7-4dae-ae84-3b2393dceb2f)  
Snake  
![Screenshot (1475)](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/4edc5f54-49e1-4fe0-817e-6c2ef98fde18)  

### chain code 
The chain codes could be generated by using conditional statements for each direction but it becomes very tedious to describe for systems having large number of directions (3-D grids can have up to 26 directions). Instead, we use a hash function. The difference in X (dx)and Y(dy) co-ordinates of two successive points are calculated and hashed to generate the key for the chain code between the two points.  
Chain code list: [5,6,7,4, -1,0,3,2,1]  
Hash function: C(dx,dy)=3dy+dx+4  
Hash table: -  
![image](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/7635d905-98a8-476b-bb68-32b35bee376a)  
The output:  
![image](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/34bcc99c-dded-4335-bca8-72c2c16d53f0)  

 #### Function to compate perimeter and area
 ```
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
```
```
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
```
 The output:  
 ![image](https://github.com/MayarFayez/CV_-Contour-Technique-/assets/93496610/d9cf86fb-0496-4fe5-b6f1-0544fd81bf40)

 






