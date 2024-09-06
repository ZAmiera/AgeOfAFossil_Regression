import pandas as pd
import streamlit as st
import pickle
import numpy as np

# Load the regression model
with open('model_pipeline.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def main():
    # Creating Sidebar for inputs
    st.sidebar.title("Inputs")
    uranium_lead_ratio = st.sidebar.slider("Uranium-Lead Ratio", 0.000241, 1.533270, 0.5)
    carbon_14_ratio = st.sidebar.slider("Carbon-14 Ratio", 0.000244, 1.000000, 0.5)
    radioactive_decay_series = st.sidebar.slider("Radioactive Decay Series", 0.000076, 1.513325, 0.5)
    stratigraphic_layer_depth = st.sidebar.slider("Stratigraphic Layer Depth (m)", 0.130000, 494.200000, 250.0)
    stratigraphic_position = st.sidebar.slider("Stratigraphic Position", 0, 2, 1)

    # Preparing input u/ pred
    inp = np.array([uranium_lead_ratio, carbon_14_ratio, radioactive_decay_series, 
                    stratigraphic_layer_depth, stratigraphic_position])
    inp = np.expand_dims(inp, axis=0)

    # Getting Pred 
    prediction = model.predict(inp)

    # Main page
    st.title("Fossil Age Prediction")
    st.write("This app predicts the age of a fossil based on geological features.")

    # Show Results when prediction is done
    if prediction.any():
        st.write('''
        ## Results
        The predicted age of the fossil is:
        ''')
        
        st.write(f"**Predicted Age: {prediction[0]:.2f} years**")

if __name__ == "__main__":
    main()
