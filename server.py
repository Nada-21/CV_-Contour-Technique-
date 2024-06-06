import streamlit as st
import cv2
from Hough import*
from ActiveContour import*

st.set_page_config(page_title=" Image Processing", page_icon="📸", layout="wide",initial_sidebar_state="collapsed")

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_st_style, unsafe_allow_html=True)

with open("style.css") as source_des:
    st.markdown(f"""<style>{source_des.read()}</style>""", unsafe_allow_html=True)
    
tab1, tab2  = st.tabs(["Hough", "ActiveContour" ])    

with tab1:
    side = st.sidebar
    uploaded_img =side.file_uploader("Upload Image",type={"png", "jpg", "jfif" , "jpeg"})
    threshold = side.number_input('Line Detection Threshold',min_value=10,max_value=300, value=60,step=1)
    col3,col4 =side.columns(2)
    minthreshold = col3.number_input('Min Edge Threshold',min_value=5,max_value=300, value=100,step=1)
    maxthreshold = col4.number_input('Max Edge Threshold',min_value=10,max_value=300, value=200,step=1)
    col5,col6 =side.columns(2)
    r_min = col5.number_input('Minimum Circle Radius',min_value=0,max_value=300, value=10,step=1)
    r_max= col6.number_input('Maximum Circle Radius',min_value=0,max_value=300, value=200,step=1)
    delta_r= side.number_input('Change Between Min & Max Radius',min_value=0,max_value=300, value=1,step=1)
    col1,col2 = st.columns(2)
    select=col2.selectbox("Select Hough",('','Line Detection','Circle Detection','Ellipse Detection'))

    if uploaded_img is not None:
        file_path = 'Images/'  +str(uploaded_img.name)
        input_img = cv2.imread(file_path)
        gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        sized_img = cv2.resize(input_img,(400,400))
        col1.image(sized_img)

        edge_image = cv2.Canny(gray_image,minthreshold, maxthreshold)
        
        if select=="Line Detection":
            output_image = line_detection(input_img,edge_image,threshold)
            output_image = cv2.resize(output_image,(450,400))
            col2.image(output_image)
        
        if select=="Circle Detection":
            output_image1 = circle_detection(input_img,edge_image,r_min,r_max,delta_r)
            output_image1 = cv2.resize(output_image1,(450,400))
            col2.image(output_image1)
        if select=="Ellipse Detection":
            output_image2 = Ellipse_detection(input_img,edge_image)
            output_image2 = cv2.resize(output_image2,(450,400))
            col2.image(output_image2)

            
with tab2:
    uploaded,activecontour1,activecontour2= st.columns(3)    
    uploadedimg =uploaded.file_uploader("Upload Image",key="tab2",type={"png", "jpg", "jfif" , "jpeg"})
    radius=uploaded.number_input('Radius',min_value=1, value=300)
    Apply=uploaded.button('Apply')
    alpha=activecontour1.number_input('Alpha', value=0.001)
    beta=activecontour1.number_input('Beta', value=0.01)
    gamma=activecontour2.number_input('Gamma', value=100)
    n=activecontour2.number_input('Num_of_iterations', value=600)
    snake=activecontour1.button('Snake')
    if uploadedimg is not None:
        file_path = 'Images/'  +str(uploadedimg.name)
        input_img = cv2.imread(file_path,0)
        fig =plt.figure(figsize=(1,1))
        plt.imshow(input_img,cmap='gray')
        plt.axis("off")
        uploaded.pyplot(fig)
        x,y=initialcontour(input_img,radius)
        if Apply:
            fig1 = plt.figure(figsize=(1,1))
            plt.imshow(input_img,cmap='gray')
            plt.plot(x,y)
            plt.axis("off")
            activecontour1.pyplot(fig1)
        if snake:
            snakes =iterate_snake(input_img,x = x,y = y,a =alpha,b =beta,gamma =gamma,n_iters =n,return_all = True)    
            fig2 = plt.figure(figsize=(1,1))
            plt.imshow(input_img,cmap='gray')
            plt.plot(np.r_[snakes[-1][0], snakes[-1][0][0]], np.r_[snakes[-1][1], snakes[-1][1][0]], c=(1,0,0), lw=2)
            plt.axis("off")
            activecontour2.pyplot(fig2)    
         
