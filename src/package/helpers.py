# --------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- All libraries, variables and functions are defined in this fil ------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

# Import sampling 
from sklearn.model_selection import train_test_split # Import the train_test_learn module
from imblearn.over_sampling import RandomOverSampler # Import the RandomOverSampler module form imbalanced-learn

# Import models:
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.ensemble import RandomForestClassifier # RandomForest
import xgboost as xgb # Extreme gradient boosting (XGBoost)
import lightgbm as lgb # LightGBM

# plotting
import plotly.graph_objs as go
from plotly.subplots import make_subplots
# 1. libraries ------------------------------------------------------------------------------------------/
# a-1) main dependencies and setup
from package.constants import * # constants

# main functions -----------------------------------------------------------------------------------------------------------------------------
# evaluation
def model_evaluation(y_test, prediction,tag, i):
    if i[1]==1:
        method_name="Original Data"
    else:
        method_name="ROS Data"
    accuracy_score = balanced_accuracy_score(y_test, prediction)
    print(f"{i[0]} - {method_name}")
    print(f"1) Accuracy Score: {round(accuracy_score,2)}%")
    print(f"------------------------------------------------------------")
    # Generate a confusion matrix for the model
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, prediction),
                                       index=["Actual "+item for item in tag],
                                       columns=["Predicted "+item for item in tag])
    print(f"2) Confusion Matrix:")
    print(confusion_matrix_df)
    print(f"------------------------------------------------------------")
    # Print the classification report for the model
    reports = classification_report(y_test, prediction, target_names=tag)
    print(f"3) Classification Report:")
    print(reports)
    
# data frame summary
def df_summary(df):
    print(f"Data Rows: {df.shape[0]}\nData Columns: {df.shape[1]}\n------------------------------------------------------------")
    summary = pd.DataFrame({'unique_count': df.nunique(),
                            'dtypes': df.dtypes,
                            'null_count': df.isnull().sum(),
                            'null(%)': 100*df.isnull().mean()})
    summary.sort_values(by='unique_count', ascending=False, inplace=True)
    print(summary)
    
    
# Plotting functions -------------------------------------------------------------------------------------------------------------------------
def sub_bar(columns, sub_name, tags, title):
    subplots_data = []
    color_plot=["#0A4853","#cd5a4d"]
    for i, data in enumerate(columns):
        subplot = go.Bar(
            name=sub_name[i],
            x=[item.upper() for item in tags],
            y=data,
            marker_color=color_plot[i],
            hovertemplate=sub_name[i].upper()+" %{x}<br>"+"<b>%{y}</b><br>"+"<extra></extra>")
        subplots_data.append(subplot)
    # Define the layout for the figure
    layout = go.Layout(
        title=dict(text="Dependent Variable",
                   font=dict(size= 24, color= 'black', family= "Times New Roman"),
                   x=0.5,
                   y=0.9),
        width=1200,
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

def sub_mix(df, hist_plot, vio_plot, title_all):
    # Create the histogram subplot
    hist_trace = go.Histogram(
        x=df[hist_plot],
        nbinsx=279,
        marker_color='#cd5a4d',
        opacity=1,
        showlegend=False
    )
    
    # Create the violin subplot
    vio_trace = [
        go.Violin(
            y=df[df[vio_plot[0]] == 0][vio_plot[1]],
            name='Healthy',
            box_visible=True,
            meanline_visible=True,
            fillcolor="#cd5a4d",
            opacity=0.7,
            line=dict(color='black', width=1),
            marker=dict(color='black', size=5, opacity=0.1),
            showlegend=False
        ),
        go.Violin(
            y=df[df[vio_plot[0]] == 1][vio_plot[1]],
            name='High Risk',
            box_visible=True,
            meanline_visible=True,
            fillcolor="#0A4853",
            opacity=0.7,
            line=dict(color='black', width=1),
            marker=dict(color='black', size=5, opacity=0.1),
            showlegend=False
        )
    ]
    
    # Define subplot titles and axis labels
    subplot_titles = [
        f"<span style='font-size: 20px; color:black; font-family:Times New Roman'>{title_all[0]}</span>",
        f"<span style='font-size: 20px; color:black; font-family:Times New Roman'>{title_all[1]}</span>"
    ]
    x_axis_labels = ["Loan Size", "Loan Status"]
    y_axis_labels = ["Frequency", "Debt to Income"]
    # Define layout
    layout = go.Layout(
        width=1200,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor= '#f7f7f7',
            font=dict(color='black')
        ),
        plot_bgcolor='#f7f7f7',
        paper_bgcolor="#ffffff"
    )

    # Create figure with subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.1
    )
    
    # Add traces to subplots
    fig.add_trace(hist_trace, row=1, col=1)
    for vio in vio_trace:
        fig.add_trace(vio, row=1, col=2)

    # Update x and y axis titles and styling
    for i in range(2):
        fig.update_xaxes(
            title=dict(
                text=x_axis_labels[i],
                font=dict(size=18, color='black', family="calibri")
            ),
            tickcolor='#ffffff',
            tickfont=dict(size=14, family='calibri', color='black'),
            showline=True,
            linewidth=0.5,
            linecolor='black',
            row=1,
            col=i+1
        )
        fig.update_yaxes(
            title=dict(
                text=y_axis_labels[i],
                font=dict(size=18, color='black', family="calibri")
            ),
            tickfont=dict(size=14, family='calibri', color='black'),
            row=1,
            col=i+1
        )

    # Update layout
    fig.update_layout(layout)

    # Display figure
    fig.show()
    
def secondary_bar(df):
    color_plot=["#0A4853","#cd5a4d"]
    fig = go.Figure(
        data=[
            go.Bar(name=df.columns[0], x=df[df.columns[0]], y=df.index,orientation='h', xaxis='x', offsetgroup=1, marker_color=color_plot[0]),
            go.Bar(name=df.columns[1], x=df[df.columns[1]], y=df.index,orientation='h', xaxis='x2', offsetgroup=2, marker_color=color_plot[1])
        ],
        layout=dict(
            xaxis=dict(title=dict(text=df.columns[0]+" importances", font=dict(size= 18, color= 'black', family= "calibri"))
                       , showline=True,linewidth=1,linecolor='black', mirror=True,
                      tickfont=dict(size= 14, family='calibri', color='black' )),
            xaxis2=dict(title=dict(text=df.columns[1]+" importances", font=dict(size= 18, color= 'black', family= "calibri"))
                        ,overlaying='x', side= 'top',
                        tickfont=dict(size= 14, family='calibri', color='black' )),
            barmode='group',
            legend=dict(yanchor="bottom",y=0.01,xanchor="right",x=0.99,bgcolor= '#f7f7f7',
            font=dict(color='black')),
            width=1200,
            height=600, 
        yaxis=dict(title=dict(text='Features', font=dict(size= 18, color= 'black', family= "calibri")),
                  tickfont=dict(size= 14, family='calibri', color='black' )),
        plot_bgcolor='#f7f7f7',
        paper_bgcolor="#ffffff")
    )

    # Change the bar mode and legend layout
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

def histogram (df, chart_title):
    trace=go.Histogram(x=df['loan_size'], nbinsx=279, marker_color='blue', opacity=0.7)
    
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
                       xaxis=dict(title='Loan Size',
                                  color= 'black',
                                  showline=True,
                                  linewidth=1,
                                  linecolor='black',
                                  mirror=True), 
                       yaxis=dict(title='Frequency',
                                  color= 'black',
                                  showline=True,
                                  linewidth=1,
                                  linecolor='black',
                                  mirror=True),
                       plot_bgcolor='#f7f7f7',
                       paper_bgcolor="#f7f7f7")
    
    fig = go.Figure(data=trace, layout=layout)
    fig.show()
# -------------------------------------------------------------------------------------------------------/
