
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title of the app
st.title("Crop Price Prediction")

# File upload feature
uploaded_file = st.file_uploader("Upload your crop dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load the CSV into a DataFrame
    crop = pd.read_csv(uploaded_file)
    
    # Display first few rows of the dataset
    st.write("Dataset preview:")
    st.dataframe(crop.head())
    
    # Display dataset info
    st.write("Dataset info:")
    buffer = io.StringIO()
    crop.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    # Display value counts for 'Price' and 'Area' columns
    st.write("Price distribution:")
    st.write(crop['Price'].value_counts())

    st.write("Area distribution:")
    st.write(crop['Area'].value_counts())
    
    # Display basic statistics
    st.write("Basic statistics:")
    st.write(crop.describe())
    
    # Display histograms of the dataset
    st.write("Histograms of numerical data:")
    fig, ax = plt.subplots(figsize=(20, 15))
    crop.hist(bins=50, ax=ax)
    st.pyplot(fig)

    # Splitting data function
    def split_train_test(data, test_ratio):
        np.random.seed(42)
        shuffled = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled[:test_set_size]
        train_indices = shuffled[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]
    
    # Split data into training and test sets
    test_ratio = st.slider("Select test data ratio:", 0.1, 0.5, 0.2)
    train_set, test_set = split_train_test(crop, test_ratio)
    
    st.write(f"Train set size: {len(train_set)}")
    st.write(f"Test set size: {len(test_set)}")

