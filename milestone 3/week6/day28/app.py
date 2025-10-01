
import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
class_names = iris.target_names

# Load model
model = joblib.load('iris_model.pkl')

st.title('🌸 Iris Flower Classification App')
st.markdown('''
This app allows you to **predict** the species of an iris flower based on its measurements, 
or **explore** the dataset visually.  
Use the sidebar to switch between modes.
''')

# Sidebar for mode
mode = st.sidebar.radio('Select Mode:', ['Prediction', 'Data Exploration'])

if mode == 'Prediction':
    st.subheader('🔮 Prediction Mode')
    # Input features with tooltips
    inputs = []
    for feature in iris.feature_names:
        val = st.slider(
            feature, 
            float(X[feature].min()), 
            float(X[feature].max()), 
            float(X[feature].mean()), 
            help=f'Select a value for {feature}'
        )
        inputs.append(val)
    inputs_df = pd.DataFrame([inputs], columns=iris.feature_names)

    prediction = model.predict(inputs_df)[0]
    probas = model.predict_proba(inputs_df)[0]

    # Color coding for predictions
    colors = {
        'setosa': 'lightgreen',
        'versicolor': 'lightblue',
        'virginica': 'lightpink'
    }
    pred_class = class_names[prediction]
    color = colors[pred_class]

    st.markdown(
        f"<div style='padding:10px; background-color:{color}; border-radius:10px;'>"
        f"<h3 style='text-align:center;'>Prediction: {pred_class.capitalize()}</h3>"
        f"</div>", 
        unsafe_allow_html=True
    )

    st.write('Prediction Probabilities:')
    st.bar_chart(probas)

elif mode == 'Data Exploration':
    st.subheader('📊 Data Exploration Mode')
    st.markdown('Select features to view their distributions and relationships.')
    # Histogram
    feature = st.selectbox('Select feature for histogram:', iris.feature_names)
    fig, ax = plt.subplots()
    sns.histplot(X[feature], kde=True, ax=ax)
    ax.set_title(f'Histogram of {feature}')
    st.pyplot(fig)

    # Scatter plot
    f1 = st.selectbox('Feature X:', iris.feature_names, index=0)
    f2 = st.selectbox('Feature Y:', iris.feature_names, index=1)
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=X[f1], y=X[f2], hue=y, palette='deep', ax=ax2)
    ax2.set_title(f'Scatter plot of {f1} vs {f2}')
    st.pyplot(fig2)
