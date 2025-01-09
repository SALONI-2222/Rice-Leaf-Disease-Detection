import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_disease_model.keras')
    #image = tf.keras.preprocessing.image.load_img(test_image,target_size=(256,256))
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(256,256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("RICE DISEASE DETECTION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Rice Disease Detection System! 🌿🔍
    
    Our mission is to help in identifying Rice plant diseases efficiently. Upload an image of a Rice plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our Rice crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Rice Disease Detection** page and upload an image of a Rice plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Rice Disease Detection** page in the sidebar to upload an image and experience the power of our Rice Disease Detection System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 5732 rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (4586 images)
                2. test (573 images)
                3. validation (573 images)

                """)



#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Rice Disease Detection")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    
     #Predict button
    if(st.button("Predict")):
        st.snow()

        st.write("Our Prediction")

        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
