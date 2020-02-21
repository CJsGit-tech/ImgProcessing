import cv2
import numpy as np
import streamlit as st
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

st.header("Image Processing Tool")
st.sidebar.header("Select Methods")


st.write("---")
# File Selecter
file = st.file_uploader("Choose an Image File",type = ['png','jpg','jpeg'])
mode = st.sidebar.radio("Choose Image Mode",["None","RGB","BGR","Gray"])
if file is not None:
    st.subheader("Image Preview")
    if mode == "RGB":
        img = Image.open(file)
        img = np.asarray(img)
        st.write("Image Shape",img.shape)
        plt.imshow(img)
        st.pyplot()
    elif mode == "BGR":
        img = Image.open(file)
        img = np.asarray(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        st.write("Image Shape",img.shape)
        plt.imshow(img)
        st.pyplot()
    elif mode =="Gray":
        img = Image.open(file)
        img = np.asarray(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        st.write("Image Shape",img.shape)
        plt.imshow(img,cmap = "gray")
        st.pyplot()
    else:
        st.info("Select Color Mode")

st.write("---")
#######################################################################
if st.sidebar.checkbox("Resize Option"):
    st.header("Resize")    
    st.sidebar.subheader("Resize")
    width = st.sidebar.number_input("Width Ratio",max_value=1.0,value = 0.5,step = 0.1)
    height = st.sidebar.number_input("Height Ratio",max_value=1.0,value = 0.5,step = 0.1)
    img = cv2.resize(img,(0,0),img,width,height)
    st.write("New Image Shape",img.shape)
    if mode =="Gray":
        plt.imshow(img,cmap = "gray")
        st.pyplot()
    else:
        plt.imshow(img)
        st.pyplot()
    st.write("---")

##################################################################
if st.sidebar.checkbox("Thrsholding Option"):
    st.header("Image Thresholding (GrayScale Only)")
    st.sidebar.subheader("Thresholding")
    thresh = st.sidebar.radio("Select Threshholding Options",["None","Binary","Binary_INV",
                                             "Truncation","TOZERO",
                                             "TOZERO_INV",
                                             "ADAPT_MEAN_BIN",
                                             "ADAPT_GAU_BIN"])
    if thresh == "Binary":
        ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        st.write("New Image Shape",img.shape)
        plt.imshow(img, cmap = "gray")
        st.pyplot()
    
    elif thresh == "Binary_INV":
        ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        st.write("New Image Shape",img.shape)
        plt.imshow(img, cmap = "gray")
        st.pyplot()
        
    elif thresh == "Truncation":
        ret, img = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
        st.write("New Image Shape",img.shape)
        plt.imshow(img, cmap = "gray")
        st.pyplot()
        
    elif thresh == "TOZERO":
        ret, img = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
        st.write("New Image Shape",img.shape)
        plt.imshow(img, cmap = "gray")
        st.pyplot()
    
    elif thresh == "TOZERO_INV":
        ret, img = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
        st.write("New Image Shape",img.shape)
        plt.imshow(img, cmap = "gray")
        st.pyplot()
        
    elif thresh == "ADAPT_MEAN_BIN":
        img = cv2.adaptiveThreshold(img,255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY,25,10) 
        st.write("New Image Shape",img.shape)
        plt.imshow(img, cmap = "gray")
        st.pyplot()
    elif thresh == "ADAPT_GAU_BIN":
        img = cv2.adaptiveThreshold(img,255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,15,8) 
        st.write("New Image Shape",img.shape)
        plt.imshow(img, cmap = "gray")
        st.pyplot()
    else:
        st.write("New Image Shape",img.shape)
        img = img
        plt.imshow(img,cmap ="gray")
        st.pyplot()
        pass
    
####################################################
st.sidebar.markdown("Blending: Coming Soon")
#####################################################
if st.sidebar.checkbox("Blurring and Smoothing"):
    st.header("Blurring and Smoothing")
    
    bluring = st.radio("Select Blurring Options",
                       ["None","Gamma","2D Filter","Averaging",
                        "Gaussian","Median","Bilateral Filtering"])
        
    if bluring == "None":
        pass
    elif bluring == "Gamma":
        st.markdown("Brightness")
        gamma = st.number_input("Gamma Value",value = 1.0)
        img = np.power(img,gamma)
        plt.imshow(img,cmap = "gray")
        st.pyplot()
    
    elif bluring == "2D Filter":
        st.markdown("2D Kernel Filter")
        shape = st.selectbox("kernel",[num for num in range(1,30,2)])
        kernel = np.ones(shape = (shape,shape),dtype = np.float32)/25
        img = cv2.filter2D(img,-1,kernel)
        plt.imshow(img,cmap = "gray")
        st.pyplot()
        
    elif bluring == "Averaging":
        st.markdown("Averaging")
        ksize = st.selectbox("Kernel Size", [num for num in range(1,30,2)])
        img = cv2.blur(img,ksize= (ksize,ksize))
        plt.imshow(img,cmap = "gray")
        st.pyplot()
    
    elif bluring == "Gaussian":
        st.markdown("Gaussian Blur")
        ksize = st.selectbox("Kernel Size", [num for num in range(1,30,2)])
        gau_const = st.number_input("Gaussian Constant",value = 1)
        img = cv2.GaussianBlur(img,(ksize,ksize),gau_const)
        plt.imshow(img,cmap = "gray")
        st.pyplot()
        
        
    elif bluring == "Median":
        st.markdown("Median Blur")
        ksize = st.number_input("Kernel Size",value = 1)
        img = cv2.medianBlur(img,ksize)
        plt.imshow(img,cmap = "gray")
        st.pyplot()
        
    elif bluring == "Bilateral Filtering":
        st.markdown("Bilateral Filtering")
        d = st.number_input("Diameter of each pixel",value = 9,min_value =0)
        sigmaColor = st.number_input("SigmaColor",value = 75,min_value = 0)
        sigmaSpace = st.number_input("SigmaSpace",value = 75,min_value = 0)
        img = cv2.bilateralFilter(img,d,sigmaColor,sigmaSpace)
        plt.imshow(img,cmap = "gray")
        st.pyplot()  
        

####################################################
if st.sidebar.checkbox("Morphological Operations"):
    st.header("Morphological Operators")
    morph = st.sidebar.radio("Select Blurring Options",
                           ["None","Erosion","Dialtion","Opening",
                            "Closing","Gradient"])
    if morph == "None":
        pass
    
    elif morph == "Erosion":
        shape = st.selectbox("kernel",[num for num in range(1,30,2)])
        kernel = np.ones(shape = (shape,shape),dtype = np.uint8)
        iterations = st.number_input("iterations",value = 1)
        img = cv2.erode(img,kernel,iterations = iterations)
        plt.imshow(img,cmap ="gray")
        st.pyplot()
    
    elif morph == "Dialtion":
        shape = st.selectbox("kernel",[num for num in range(1,30,2)])
        kernel = np.ones(shape = (shape,shape),dtype = np.uint8)
        iterations = st.number_input("iterations",value = 1)
        img = cv2.dilate(img,kernel,iterations = iterations)
        plt.imshow(img,cmap ="gray")
        st.pyplot()
    
    elif morph == "Opening":
        shape = st.selectbox("kernel",[num for num in range(1,30,2)])
        kernel = np.ones(shape = (shape,shape),dtype = np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        plt.imshow(img,cmap ="gray")
        st.pyplot()
    
    elif morph == "Closing":
        shape = st.selectbox("kernel",[num for num in range(1,30,2)])
        kernel = np.ones(shape = (shape,shape),dtype = np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        plt.imshow(img,cmap ="gray")
        st.pyplot()
    
    elif morph == "Gradient":
        shape = st.selectbox("kernel",[num for num in range(1,30,2)])
        kernel = np.ones(shape = (shape,shape),dtype = np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        plt.imshow(img,cmap ="gray")
        st.pyplot()
    
##################################################
if st.sidebar.checkbox("Gradients ; Emphasis"):
    st.header("Gradients")
    graident = st.sidebar.radio("Select Blurring Options",
                               ["None","SobelX","SobelY","Laplacian"])
    
    if graident == "None":
        pass
    elif graident == "SobelX":
        ksize = st.number_input("Kernel Size",value = 3,step =2,max_value = 31)
        img = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=ksize)
        plt.imshow(img,cmap ="gray")
        st.pyplot()
        
    elif graident =="SobelY":
        ksize = st.number_input("Kernel Size",value = 3,step = 2,max_value = 31)
        img = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=ksize)
        plt.imshow(img,cmap ="gray")
        st.pyplot()
        
    elif graident == "Laplacian":
        img = cv2.Laplacian(img,cv2.CV_64F)
        plt.imshow(img,cmap ="gray")
        st.pyplot()    
##################################################3
        
        
st.header("Save File")
folder = st.text_input("Input Folder Name")
name = st.text_input("Input File Name")
if st.button("save"):
    try:
        os.mkdir("{}".format(folder))
        cv2.imwrite("{}/{}.png".format(folder,name),img)
    except FileExistsError:
        st.warning("Folder Exists")

        