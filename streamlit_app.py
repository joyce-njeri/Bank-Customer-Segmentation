import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler


def load_model():
    with open('kmeans_saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

clusterer = data["model"]
data = data["data"]
results = data.copy()

st.write(
    """
    # Bank Customer Segmentation

    A Machine Learning App that performs customer segmentation and 
    identifies popular customer groups along with their definitions
    and recommendations.

    """
)

clusters = (
    "cluster_2",
    "cluster_3",
    "cluster_4",
    "cluster_5",
    "cluster_6",
    "cluster_7",
    "cluster_8",
)

cols = [
    'AccountBalance', 
    'TransactionAmount', 
    'TransactionMonth',
    'TransactionDay', 
    'TransactionWeek', 
    'TransactionDayofweek', 
    'Recency',
    'TransactionFrequency', 
    'CustomerAge_atTxn',
    'Transaction_percentage_of_Balance', 
    'TransactionHour',
    'TransactionSeconds',
]

scores = {
    "2":0.19,
    "3":0.17,
    "4":0.16,
    "5":0.14,
    "6":0.13,
    "7":0.12,
    "8":0.11,
}

## Data upload
st.sidebar.markdown('## Import Dataset') 
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv") 

## display feature profiles
def profile_feature(df_profile, feature, clusters):
    #Checks if it's a binary 
    if df_profile[feature].nunique() > 2:
        #If not binary, make Box plots
        box_data = [go.Box(y=df_profile.loc[df_profile[clusters] == k, feature].values, name=f'Cluster {k+1}') for k in np.unique(df_profile[clusters])]
        fig = go.Figure(data=box_data)
    else:
        #If binary, make bar plot
        x =[f'Cluster {k}' for k in np.unique(df_profile[clusters])]
        y = [df_profile.loc[df_profile[clusters] == k, feature].mean() for k in np.unique(df_profile[clusters])]
        fig = go.Figure([go.Bar(x=x, y=y)])

    return fig

## Data Export
def get_table_download_link(df, filename, linkname):

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{linkname}</a>'
    return href

def profile_clusters(df, cols, clusters):
  cluster_names = [f'Cluster {k+1}' for k in np.unique(df[clusters])] # X values such as "Cluster 1", "Cluster 2", etc
  data = [go.Bar(name=f, x=cluster_names, y=df.groupby(clusters)[f].mean()) for f in cols] #a list of plotly GO objects with different Y values
  fig = go.Figure(data=data)
  fig.update_layout(barmode='group')

  return fig

def profiles(df, cluster_col):
    df = df.groupby([cluster_col]).mean().round(1)
    return df

feat = False
prof = False
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(f'<h3 "> Data Preview </h3>',unsafe_allow_html=True)
    st.write(df.head())

    feat = st.checkbox('Check to explore clusters')

else:
    st.write("Please upload your CSV file to begin clustering")

## display cluster profiles
if (feat == True) & (uploaded_file is not None):
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(f'<h3 "> Features Overview </h3>',unsafe_allow_html=True)
    feature = st.selectbox('Select a feature...', options = cols)
    if (feature == ""):
        feature = 'AccountBalance'
    st.sidebar.markdown('## Explore clusters') 
    cluster_col = st.sidebar.selectbox('Pick your cluster', options=clusters)
    if (cluster_col == ""):
        cluster_col = 'cluster_2'
    feat_fig = profile_feature(data, feature, cluster_col)
    st.plotly_chart(feat_fig)

    prof = st.checkbox('Check to profile the clusters')

elif (feat == False) & (uploaded_file is not None):
    st.markdown("""
    <br>
    <h2> Please check the box to profile customers</h2>

    """, unsafe_allow_html=True) 

if (prof == True) & (uploaded_file is not None):
    # Scaling for plotting
    for c in cols:
        data[c] = MinMaxScaler().fit_transform(np.array(data[c]).reshape(-1, 1))

    st.sidebar.markdown('## Explore feature groups')
    features_col = st.sidebar.multiselect('Pick your features', options=[c for c in cols], default = cols)
    if (features_col == ""):
        features_col = cols

    fig = profile_clusters(data, features_col, cluster_col)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(f'<h3 "> Profiles for {clusters.index(cluster_col)+2} Clusters </h3>',
                            unsafe_allow_html=True)
    fig.update_layout(
        autosize=True,
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=40,
                pad=0
            ),
        )
    st.plotly_chart(fig)

    show = st.checkbox('Show clusters performance scores (silhouette)')
    if show == True:
        st.write(scores)

    df = profiles(results, cluster_col)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('### Profiles Preview')
    st.write(df[cols].head(10))

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('### Downloads')
    st.write(get_table_download_link(results,'profiles.csv', 'Download profiles'), unsafe_allow_html=True)