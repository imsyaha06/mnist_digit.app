# import streamlit as st
# import numpy as np
# import cv2
# from keras.models import load_model
# from streamlit_drawable_canvas import st_canvas

# # Load the saved model
# model = load_model('mnist_ann_model.h5')

# # Function to preprocess the drawn image
# def preprocess_image(image):
#     # Resize the image to 28x28 pixels
#     image = cv2.resize(image, (28, 28))

#     # Convert the image to grayscale
#     image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

#     # Normalize the image
#     image = image / 255.0

#     # Expand dimensions to match the input shape of the model (1, 28, 28)
#     image = np.expand_dims(image, axis=0)

#     return image

# # Streamlit app title
# st.title("MNIST Digit Recognition")

# # Create a drawable canvas
# canvas_result = st_canvas(
#     fill_color="black",
#     stroke_width=25,
#     stroke_color="white",
#     background_color="black",
#     height=280,
#     width=280,
#     drawing_mode="freedraw",
#     key="canvas",
# )

# # Button to predict the digit
# if st.button("Predict"):
#     if canvas_result.image_data is not None:
#         # Preprocess the image
#         img = preprocess_image(canvas_result.image_data)
        
#         # Make prediction
#         predictions = model.predict(img)
#         predicted_class = np.argmax(predictions)

#         # Show the predicted digit
#         st.write(f"Predicted Digit: {predicted_class}")

#         # Show the drawn image
#         st.image(canvas_result.image_data, caption="Drawn Digit", use_column_width=True)
#     else:
#         st.write("Please draw a digit first.")













# !pip install streamlit --quiet
# !pip install pyngrok==4.1.1 --quiet
# !pip install streamlit-drawable-canvas --quiet
# from pyngrok import ngrok





 
# import streamlit as st
# from streamlit_drawable_canvas import st_canvas
# from tensorflow import keras
# import cv2
# import numpy as np
# model_new = keras.models.load_model('/content/drive/MyDrive/Minor Project /mnist.hdf5')

# st.title("MNIST Digit Recognizer")

# SIZE = 192

# canvas_result = st_canvas(
#     fill_color="#ffffff",
#     stroke_width=10,
#     stroke_color='#ffffff',
#     background_color="#000000",
#     height=150,width=150,
#     drawing_mode='freedraw',
#     key="canvas",
# )

# if canvas_result.image_data is not None:
#     img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
#     img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
#     st.write('Input Image')
#     st.image(img_rescaling)

# if st.button('Predict'):
#     test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     pred = model_new.predict(test_x.reshape(1, 28, 28, 1))
#     st.write(f'result: {np.argmax(pred[0])}')
#     st.bar_chart(pred[0])

     
 

 
# url = ngrok.connect(port='8501')
# url





import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Streamlit app layout
st.title("MNIST Digit Generator")

# User input: type a digit between 0 and 9
digit = st.number_input("Enter a digit (0-9):", min_value=0, max_value=9, step=1)

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Button to display the image
if st.button("Show Image"):
    try:
        # Find an image from the test set corresponding to the input digit
        digit_indices = np.where(y_test == digit)[0]
        
        if len(digit_indices) > 0:
            # Select a random image for the chosen digit
            random_index = np.random.choice(digit_indices)
            selected_image = X_test[random_index]
            
            # Display the image
            st.write(f"Image of digit: {digit}")
            st.image(selected_image, width=100)
        else:
            st.error("No images found for the given digit.")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
