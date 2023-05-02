# --------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- All libraries, variables and functions are defined in this file -----------------------------------
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

from package.constants import * # constants

# main functions
# --------------------------------------------------------------------------------------------------------------------------------------------
# data frame summary _________________________________________________________________________________________________________________________
def df_summary(df):
    """
    Data summary contain number of rows and columns, unique count, datatype and number of null
    
    Args:
        df: a pandas DataFrame
    Returns:
        dataframe summry report.    
    """
    print(f"Data Rows: {df.shape[0]}\nData Columns: {df.shape[1]}{H_LINE_ENTER}")
    summary = pd.DataFrame({"unique_count": df.nunique(),
                            "dtypes": df.dtypes,
                            "null_count": df.isnull().sum(),
                            "null(%)": 100*df.isnull().mean()})
    summary.sort_values(by="unique_count", ascending=False, inplace=True)
    return summary

# model evaluation ___________________________________________________________________________________________________________________________
def model_evaluation(y_test, prediction, i):
    """
    Evaluates the performance of a classification model by computing its accuracy score,
    generating a confusion matrix, and printing a classification report.

    Args:
        y_test: a pandas DataFrame that represents the true labels of the test set.
        prediction:a numpy array that contains the predicted labels of the test set.
        i: a list that contains an "methods name" and number 1 or 2.

    Returns:
        A tuple containing the accuracy score, confusion matrix, and classification report.
    """
    if i[1]==1:
        method_name= DATA_ID[0]
    else:
        method_name= DATA_ID[1]
    # Generate and print accuracy score for the model
    accuracy_score = balanced_accuracy_score(y_test, prediction)
    print(f"{i[0]} - {method_name}")
    print(f"1) Accuracy Score: {round(accuracy_score,2)}%")
    print(f"{H_LINE}")
    # Generate and print confusion matrix for the model
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, prediction),
                                       index=["Actual "+TAG for TAG in TAGS],
                                       columns=["Predicted "+TAG for TAG in TAGS])
    print(f"2) Confusion Matrix:")
    print(confusion_matrix_df)
    print(f"{H_LINE}")
    # Generate and print the classification report for the model
    reports = classification_report(y_test, prediction, target_names=TAGS)
    print(f"3) Classification Report:")
    print(reports)
    return accuracy_score, confusion_matrix_df, reports
      
# Plotting functions
# --------------------------------------------------------------------------------------------------------------------------------------------
# histogram and violin plot mix ______________________________________________________________________________________________________________
def sub_mix(df, hist_plot, vio_plot):
    """
    plotting mix histogram and violin plots for showing colums frequency and statistic summary.

    Args:
        df: a pandas DataFrame.
        hist_plot: the column to use for the histogram.
        vio_plot: and the two columns to use for the violin plots.

    Returns:
        Displays the plot.
    """
    color_plot=COLOR_SET
    # Create the histogram subplot
    hist_trace = go.Histogram(
        x=df[hist_plot],
        nbinsx=279,
        marker_color= color_plot[1],
        hovertemplate= hist_plot.replace('_', ' ').upper()+": <b>%{x}</b><br>"+"FREQUENCY: <b>%{y}</b>"+"<extra></extra>",
        opacity=1,
        showlegend=False
    )
    
    # Create the violin subplot
    vio_trace = []
    for i, TAG in enumerate(TAGS):
        vio_trace.append(
            go.Violin(
                y=df[df[vio_plot[0]] == i][vio_plot[1]],
                name=TAG.upper(),
                box_visible=True,
                meanline_visible=True,
                fillcolor=color_plot[i],
                opacity=0.7,
                line=dict(color="black", width=1),
                marker=dict(color="black", size=5, opacity=0.1),
                showlegend=False
            )
        )
    
    # Define subplot titles and axis labels
    subplot_titles = [
        f"<span style='font-size: 20px; color:black; font-family:Times New Roman'>Distribution of {hist_plot.replace('_', ' ')}</span>",
        f"<span style='font-size: 20px; color:black; font-family:Times New Roman'>{vio_plot[1].replace('_', ' ').capitalize()} by {vio_plot[0].replace('_', ' ')}</span>"
    ]
    
    x_axis_labels = [hist_plot.replace('_', ' ').capitalize(), vio_plot[0].replace('_', ' ').capitalize()]
    y_axis_labels = ["Frequency", vio_plot[1].replace('_', ' ').capitalize()]
    # Define layout
    layout = go.Layout(
        width=PLT_WIDTH,
        height=PLT_HEIGHT,
        legend=PLT_LEGEND,
        plot_bgcolor=PLT_BGCOLOR,
        paper_bgcolor=PLT_PAPER_BGCOLOR
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
                font=AXES_TITLE_FONT
            ),
            tickfont=TICK_FONT,
            showline=True,
            linewidth=0.5,
            linecolor="black",
            row=1,
            col=i+1
        )
        fig.update_yaxes(
            title=dict(
                text=y_axis_labels[i],
                font=AXES_TITLE_FONT
            ),
            tickfont=TICK_FONT,
            row=1,
            col=i+1
        )

    # Update layout
    fig.update_layout(layout)

    # Display figure
    return fig.show()
    
# sub bar ____________________________________________________________________________________________________________________________________
def sub_bar(columns, sub_name, plt_title):
    """
    This function creates a sequential bar chart using the data in "columns" and "sub_name" and displays the plot.
    
    Args:
        columns: A numpy array containing the data for the bar chart.
        sub_name: A list containing the name of each subplot.
        plt_title: The title for the plot.

    Returns:
        Displays the plot.
    """
    subplots_data = []
    color_plot=COLOR_SET
    for i, data in enumerate(columns):
        subplot = go.Bar(
            name=sub_name[i],
            x=[TAG.upper() for TAG in TAGS],
            y=data,
            marker_color=color_plot[i],
            hovertemplate=sub_name[i].upper()+" %{x}<br>"+"<b>%{y}</b><br>"+"<extra></extra>")
        subplots_data.append(subplot)
        
    # Define the layout for the figure
    layout = go.Layout(
        title=dict(text=plt_title,
                   font=PLT_TITLE_FONT,
                   x=0.5,
                   y=0.9),
        width=PLT_WIDTH,
        height=PLT_HEIGHT,
        legend=PLT_LEGEND,
        xaxis=dict(color= "black",
                   showline=True,
                   linewidth=1,
                   linecolor="black"), 
        yaxis=dict(title=dict(text="Counts", font=AXES_TITLE_FONT)),
        plot_bgcolor=PLT_BGCOLOR,
        paper_bgcolor=PLT_PAPER_BGCOLOR
    )
    # Create the figure with two subplots
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.1)

    # Add the subplots to the figure
    for i in range(2):
        fig.add_trace(subplots_data[i], row=1, col=i+1)
        fig.update_xaxes(
            title=dict(
                text=sub_name[i],
                font=AXES_TITLE_FONT
            ),
            tickfont=TICK_FONT,
            showline=True,
            linewidth=0.5,
            linecolor="black",
            row=1,
            col=i+1
        )
        fig.update_yaxes(
            tickfont=TICK_FONT,
            row=1,
            col=i+1
        )

    # Update the layout of the figure
    fig.update_layout(layout, showlegend=False)

    # Show the figure
    return fig.show()

# two axes bar chart ________________________________________________________________________________________________________________________ 
def secondary_bar(df):
    """
    This function creates a bar chart with secondry x-axes using dataframe and displays the plot.
    
    Args:
        df: a pandas DataFrame.

    Returns:
        Displays the plot.
    """
    color_plot=COLOR_SET
    fig = go.Figure(
        data=[
            go.Bar(name=df.columns[0], x=df[df.columns[0]],
                   y=df.index,orientation="h", xaxis="x", offsetgroup=1, marker_color=color_plot[0],
                   hovertemplate= "Features: <b>%{y}</b><br>"+"Importances: <b>%{x}</b>"),
            go.Bar(name=df.columns[1], x=df[df.columns[1]],
                   y=df.index,orientation="h", xaxis="x2", offsetgroup=2, marker_color=color_plot[1],
                   hovertemplate= "Features: <b>%{y}</b><br>"+"Importances: <b>%{x}</b>")
        ],
        layout=dict(
            xaxis=dict(
                title=dict(
                    text=df.columns[0]+" importances",
                    font=AXES_TITLE_FONT
                ),
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                tickfont=TICK_FONT
            ),
            xaxis2=dict(
                title=dict(
                    text=df.columns[1]+" importances",
                    font=AXES_TITLE_FONT
                ),
                overlaying="x",
                side= "top",
                tickfont=TICK_FONT
            ),
            barmode="group",
            legend=PLT_LEGEND_BOTTOM,
            width=PLT_WIDTH,
            height=PLT_HEIGHT, 
            yaxis=dict(
                title=dict(
                    text="Features",
                    font=AXES_TITLE_FONT),
                tickfont=TICK_FONT),
            plot_bgcolor=PLT_BGCOLOR,
            paper_bgcolor=PLT_PAPER_BGCOLOR
        )
    )

    # Change the bar mode and legend layout
    return fig.show()
    
# --------------------------------------------------------------------------------------------------------------------------------------------
