import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Streamlit app layout
st.title("MNIST Digit Generator by ANN")

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
