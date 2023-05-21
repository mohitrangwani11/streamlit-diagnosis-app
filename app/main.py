import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import pickle
import numpy as np

## in the real world scenario we would not import data directly in the app. 

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32','id'],axis = 1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    return data

def add_sidebar():

    #handles the sliders and converts it into a dictironary of inputs for gathering user modified values

    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    slider_labels = [
    ("Radius (mean)", "radius_mean"),
    ("Texture (mean)", "texture_mean"),
    ("Perimeter (mean)", "perimeter_mean"),
    ("Area (mean)", "area_mean"),
    ("Smoothness (mean)", "smoothness_mean"),
    ("Compactness (mean)", "compactness_mean"),
    ("Concavity (mean)", "concavity_mean"),
    ("Concave Points (mean)", "concave points_mean"),
    ("Symmetry (mean)", "symmetry_mean"),
    ("Fractal Dimension (mean)", "fractal_dimension_mean"),
    ("Radius (se)", "radius_se"),
    ("Texture (se)", "texture_se"),
    ("Perimeter (se)", "perimeter_se"),
    ("Area (se)", "area_se"),
    ("Smoothness (se)", "smoothness_se"),
    ("Compactness (se)", "compactness_se"),
    ("Concavity (se)", "concavity_se"),
    ("Concave Points (se)", "concave points_se"),
    ("Symmetry (se)", "symmetry_se"),
    ("Fractal Dimension (se)", "fractal_dimension_se"),
    ("Radius (worst)", "radius_worst"),
    ("Texture (worst)", "texture_worst"),
    ("Perimeter (worst)", "perimeter_worst"),
    ("Area (worst)", "area_worst"),
    ("Smoothness (worst)", "smoothness_worst"),
    ("Compactness (worst)", "compactness_worst"),
    ("Concavity (worst)", "concavity_worst"),
    ("Concave Points (worst)", "concave points_worst"),
    ("Symmetry (worst)", "symmetry_worst"),
    ("Fractal Dimension (worst)", "fractal_dimension_worst")
]
    
    input_dict = {}
    
    for label, key in  slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value = float(0), 
            max_value = float(data[key].max()),
            value = float(data[key].mean())
        )   
    return input_dict

def get_scaled_values(input_dict):
    #bachpan wala scaling, should have used scikit learn here :/ 
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis = 1)

    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val)/ (max_val - min_val)
        scaled_dict[key] = scaled_value


    return scaled_dict

def get_radar_chart(input_data):
    ## used to plot the graph 

    input_data = get_scaled_values(input_data)
    categories = ['Area', 'Compactness', 'Concave Points', 'Concavity', 'Fractal Dimension', 'Perimeter', 'Radius', 'Smoothness', 'Symmetry', 'Texture']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
      r=[input_data['area_mean'], input_data['compactness_mean'], input_data['concave points_mean'],
         input_data['concavity_mean'], input_data['fractal_dimension_mean'], input_data['perimeter_mean'],
         input_data['radius_mean'], input_data['smoothness_mean'], input_data['symmetry_mean'],input_data['texture_mean']
         ],
      theta=categories,
      fill='toself',
      name='Mean Values'
      ))
    
    fig.add_trace(go.Scatterpolar(
      r=[input_data['area_se'], input_data['compactness_se'], input_data['concave points_se'], input_data['concavity_se'], 
         input_data['fractal_dimension_se'], input_data['perimeter_se'], input_data['radius_se'], input_data['smoothness_se'], 
         input_data['symmetry_se'], input_data['texture_se']
         ],
      theta=categories,
      fill='toself',
      name='Standard Error'))
    
    fig.add_trace(go.Scatterpolar(
      r=[input_data['area_worst'], input_data['compactness_worst'], input_data['concave points_worst'], 
         input_data['concavity_worst'], input_data['fractal_dimension_worst'], input_data['perimeter_worst'], 
         input_data['radius_worst'], input_data['smoothness_worst'], input_data['symmetry_worst'], input_data['texture_worst']
         ],
      theta=categories,
      fill='toself',
      name='Worst Value'))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
        showlegend=True)
    return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl","rb"))
    scaler = pickle.load(open("model/scaler.pkl","rb"))

    input_array = np.array(list(input_data.values())).reshape(1,-1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write(" The cell cluster is: ")

    if prediction[0] == 0:
        st.header(':green[Benign]')
    else: 
        st.header(':red[Malignant]')

    st.write(" The probability of being Benign is: ", model.predict_proba(input_array_scaled)[0][0])
    st.write(" The probability of being Malignant is: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("This app is only intended to guide the medical process. This is not be considered as a diagnosis in itself.")
    


    return 1
     
def main():
    #print('Hello World, Streamlit app')
    st.set_page_config(
        page_title= "Cancer Predictor",
        page_icon=":female-doctor:",
        layout = "wide",
        initial_sidebar_state="expanded"
    )
    #st.write("Hello World")

    # with open("assets/style.css") as f:
    #     st.markdown("<stlye>{}</style>".format(f.read()), unsafe_allow_html=True)


    input_data = add_sidebar()
    #st.write(input_data) # just to test if it actually captures the data

    with st.container():
        st.title("Cancer Predictor")
        st.write("Connect this application to your simulation app to help diagnose the cancer form the tissue sample")

    col1, col2 = st.columns([4,1]) # sets the ratio of the columns

    with col1:
        #st.write("this is column 1")
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        #st.write("this is column 2")
        add_predictions(input_data)


if __name__ =='__main__':
    main()
