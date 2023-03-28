# --------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- All libraries, variables and functions are defined in this fil ------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------



import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

# Import Models:
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.ensemble import RandomForestClassifier # RandomForest


from imblearn.over_sampling import RandomOverSampler # Import the RandomOverSampler module form imbalanced-learn


# ------------------------------------------------
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Import confusion_matrix
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_curve, f1_score, auc
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,roc_auc_score
from sklearn.metrics import accuracy_score,log_loss
from sklearn import tree, metrics
# ---------------------------------------------------

import pandas as pd

import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

from sklearn.cluster import KMeans
# plotting
import plotly.graph_objs as go
from plotly.subplots import make_subplots
# 1. libraries ------------------------------------------------------------------------------------------/
# a-1) main dependencies and setup
from package.constants import * # constants

# main functions -----------------------------------------------------------------------------------------------------------------------------
def model_evaluation(prediction):
    accuracy_score = balanced_accuracy_score(y_test, prediction)
    print(f"Logistic Regression Model - Original Data")
    print(f"1) Accuracy Score: {round(accuracy_score,2)}%")
    print(f"------------------------------------------------------------")
    # Generate a confusion matrix for the model
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, prediction),
                                       index=["Actual "+item for item in target_names],
                                       columns=["Predicted "+item for item in target_names])
    print(f"2) Confusion Matrix:")
    print(confusion_matrix_df)
    print(f"------------------------------------------------------------")
    # Print the classification report for the model
    reports = classification_report(y_test, prediction, target_names=target_names)
    print(f"3) Classification Report:")
    print(reports)
    
# Plotting functions -------------------------------------------------------------------------------------------------------------------------
def sub_bar(columns, sub_name, tags, title):
    subplots_data = []
    for i, data in enumerate(columns):
        subplot = go.Bar(
            name=sub_name[i],
            x=[item.upper() for item in tags],
            y=data,
            hovertemplate=sub_name[i].upper()+" %{x}<br>"+"<b>%{y}</b><br>"+"<extra></extra>")
        subplots_data.append(subplot)
    # Define the layout for the figure
    layout = go.Layout(
        title=dict(text="Dependent Variable",
                   font=dict(size= 24, color= 'black', family= "Times New Roman"),
                   x=0.5,
                   y=0.9),
        width=1000,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor= '#f7f7f7',
            font=dict(color='black')),
        xaxis=dict(color= 'black',
                   showline=True,
                   linewidth=1,
                   linecolor='black'), 
        yaxis=dict(title=dict(text='Counts', font=dict(size= 14, color= 'black', family= "calibri"))),
        plot_bgcolor='#f7f7f7',
        paper_bgcolor="#ffffff"
    )
    # Create the figure with two subplots
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.1)

    # Add the subplots to the figure
    for i in range(2):
        fig.add_trace(subplots_data[i], row=1, col=i+1)
        fig.update_xaxes(title=dict(text=sub_name[i], font=dict(size= 18, color= 'black', family= "Times New Roman")),
                         tickcolor='#ffffff',
                tickfont=dict(size= 14, family='calibri', color='black' ),
                                         showline=True, linewidth=0.5, linecolor='black', row=1, col=i+1)
        fig.update_yaxes(tickfont=dict(size= 14, family='calibri', color='black' ),
                         row=1, col=i+1)


        # Update the layout of the figure
    fig.update_layout(layout, showlegend=False)


    # Show the figure
    fig.show()
    
def line (df, chart_title):
    # Create a list of traces for each column in the DataFrame
    traces = []
    for i, col in enumerate(df.columns):
        col_name = col.split("_")
        trace = go.Scatter(x=df.index,
                           y=df[col],
                           name=col_name[-1],
                           mode='lines',
                           line=dict(color=SEVENSET[i%len(SEVENSET)]),
                          )
        traces.append(trace)
    # Create the layout
    layout = go.Layout(title=dict(text=chart_title,
                                  font=dict(size= 24, color= 'black', family= "Times New Roman"),
                                  x=0.5,
                                  y=0.9),
                       width=1000,
                       height=600,
                       legend=dict(
                           yanchor="top",
                           y=0.99,
                           xanchor="left",
                           x=0.01,
                           bgcolor= '#f7f7f7',
                           font=dict(color='black')),
                       xaxis=dict(title='Crypto',
                                  color= 'black',
                                  showline=True,
                                  linewidth=1,
                                  linecolor='black',
                                  mirror=True), 
                       yaxis=dict(title='Price Change (%)',
                                  color= 'black',
                                  showline=True,
                                  linewidth=1,
                                  linecolor='black',
                                  mirror=True),
                       plot_bgcolor='#f7f7f7',
                       paper_bgcolor="#f7f7f7")

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)
    # Show the figure
    fig.show()

def histogram (df, bins, location):
    # Set the figure size
    plt.figure(figsize=(FIG_HEIGHT, FIG_WIDTH))

    #Plot the Clusters
    ax = sns.scatterplot(data = df_market_scaled,
                         x = 'price_change_percentage_24h',
                         y = 'price_change_percentage_7d', 
                         hue = km.labels_, 
                         palette = 'colorblind', 
                         alpha = 0.8, 
                         s = 150,
                         legend = False)

    #Plot the Centroids
    ax = sns.scatterplot(data = cluster_centers, 
                         x = 'price_change_percentage_24h',
                         y = 'price_change_percentage_7d', 
                         hue = cluster_centers.index, 
                         palette = 'colorblind', 
                         s = 600,
                         marker = 'D',
                         ec = 'black', 
                         legend = False)

    # Add Centroid Labels
    for i in range(len(cluster_centers)):
                   plt.text(x = cluster_centers.price_change_percentage_24h[i], 
                            y = cluster_centers.price_change_percentage_7d[i],
                            s = i, 
                            horizontalalignment='center',
                            verticalalignment='center',
                            size = 15,
                            weight = 'bold',
                            color = 'white')
# -------------------------------------------------------------------------------------------------------/
