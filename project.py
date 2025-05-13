# Created on May 8th
# Chart 1 (Time Series) and Chart 2 (Distribution) functionality, with comments.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# --------------------------------------------------------------------------------
# Load and preprocess the ILI data
# Uses Streamlit's caching to speed up data loading on reruns.
@st.cache_data
def load_data():

    # Read the data from the "ilidata.csv" file into a pandas DataFrame.
    # A DataFrame is like a table, good for handling structured data.
    df = pd.read_csv("ilidata.csv")

    # Remove rows where the 'ili' column has missing values (NaN).
    # We only want to work with records that have valid ILI data.
    df = df.dropna(subset=["ili"])

    # Column counts weeks starting from 0 for each state, after sorting.
    # Used as the x-axis for the time series chart.
    df["weeks"] = df.groupby("state").cumcount()

    return df

df_loaded = load_data()

# --------------------------------------------------------------------------------
# Dictionary to map state abbreviations to full state names
us_state_names = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
}

# Set the main title for the Streamlit web application.
st.title("Influenza-Like Illness (ILI) Trends, 2010 – 2025")

# Create a dropdown select box to choose a state.
state = st.selectbox(
    label="Choose a U.S. State:",
    options=sorted(df_loaded["state"].unique()),
    format_func=lambda x: us_state_names.get(x.upper(), x.upper())
)

# Get data only for the state selected.
state_df = df_loaded[df_loaded["state"] == state].copy()

# Get the full name of the selected state.
full_state = us_state_names.get(state.upper(), state.upper())



# --- Chart 1: Weekly Percent ILI Time Series ---
st.subheader(f"Weekly Percent ILI in {full_state}")

# Create a checkbox to let the user toggle the 5-week rolling mean.
# 'average' will be True if checked (default), False otherwise.
average = st.checkbox("Show 5-Week Rolling Mean", value=True)

# Set the 'weeks' column (cumulative week count, 0 to N) as the index for the x-axis.
# Select the 'ili' column for the y-axis.
y_series_data = state_df.set_index("weeks")["ili"]

# Apply rolling mean if the checkbox is selected.
if average:
    if not y_series_data.empty:
        # Apply rolling mean. Values are taken, rolled, then re-indexed with original epiweek strings.
        rolled_values = pd.Series(y_series_data.values).rolling(window=5, min_periods=1).mean()
        y_series = pd.Series(rolled_values.values, index=y_series_data.index, name="ili")
    else:
        y_series = y_series_data # an empty series
else:
    y_series = y_series_data

st.line_chart(
    y_series,
    x_label="Week Number",
    y_label="% ILI"
)



# --- Chart 2: Distribution of Weekly ILI % ---
st.subheader(f"Distribution of Weekly ILI % in {full_state}")

# Extract the 'ili' values for the selected state as a NumPy array.
ili_value = state_df["ili"].values

# - ili_value.size > 0: Ensures the array is not empty.
# - ili_value.mean() > 0: Ensures the mean is positive, as lambda_hat = 1/mean.
if ili_value.size > 0 and ili_value.mean() > 0:
    # Estimate the lambda parameter for the exponential distribution.
    # Based on LLN, sample mean (ili_value.mean()) is an estimate of the true mean of the distribution.
    # For an exponential distribution, mean = 1/lambda.
    # So, lambda_hat = 1 / sample_mean.
    lambda_hat = 1.0 / ili_value.mean()

    # Create a matplotlib figure and an axes object for plotting.
    fig, ax = plt.subplots()

    # Plot the histogram of the observed ILI percentages.
    # - ili_value: The data.
    # - bins=30: Number of bars in the histogram.
    # - density=True: Normalizes the histogram so its area sums to 1 (for comparison with PDF).
    # - alpha=0.6: Sets transparency of bars.
    # - label: Name for the legend.
    ax.hist(ili_value, bins=30, density=True, alpha=0.6, label="Observed ILI %")
    
    # Prepare x-values for plotting the exponential PDF curve.
    # Extends slightly beyond the max observed ILI value for better visualization.
    x_max_plot = ili_value.max() * 1.1
    # Generate 300 evenly spaced points from 0 to x_max_plot.
    x_vals_for_pdf = np.linspace(0, x_max_plot, 300)

    # Plot the PDF of the estimated exponential distribution.
    # - x_vals_for_pdf: X-coordinates for the curve.
    # - expon(scale=1/lambda_hat).pdf(x_vals_for_pdf): Calculates PDF values.
    #   'scale' parameter for scipy's expon is 1/lambda.
    # - 'r-': Plot as a red solid line.
    # - lw=2: Line width.
    # - label: Name for the legend, showing the estimated lambda_hat.
    ax.plot(x_vals_for_pdf, expon(scale=1/lambda_hat).pdf(x_vals_for_pdf), 'r-', lw=2, label=f"Exponential PDF ($\\hat{{\\lambda}} \\approx {lambda_hat:.2f}$)")
    
    ax.set_xlabel("Weekly ILI Percentage (%)")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)



    # --- Interpretation Text ---
    st.header("Interpretation of Charts")
    st.markdown(
        f"""
        **Time Series: Weekly Percent ILI in {full_state}**

        The line chart above displays the percentage of outpatient visits attributed to Influenza-Like Illness (ILI)
        in **{full_state}** over time (2010-2025). The x-axis, "Week Number," represents the number of weeks elapsed since the
        earliest data point for this state in the dataset (starting from week 0). The y-axis shows the ILI percentage.
        You can use the checkbox to toggle a 5-week rolling average, which helps to smooth out
        short-term fluctuations and highlight underlying seasonal trends in ILI activity.

        **Distribution: Weekly ILI % in {full_state}**

        The histogram shows the overall distribution of the weekly ILI percentages observed in **{full_state}**.
        The x-axis is the ILI percentage, and the y-axis ("Density") indicates how common different ranges of ILI percentages are
        (normalized so the total area is 1).
        The overlaid red curve is the probability density function (PDF) of an Exponential distribution that has been
        fitted to the ILI data. The parameter for this exponential distribution, $\\lambda$ (lambda), is estimated
        using the average of the observed weekly ILI percentages.

        **Why an Exponential Distribution? (Law of Large Numbers Connection)**

        The Law of Large Numbers (LLN) suggests that if we have many independent observations from the same underlying
        distribution, their sample average will be a good estimate of that distribution's true mean.
        For an Exponential distribution, the theoretical mean is $1/\\lambda$.
        By calculating the average of all observed weekly ILI values ($\overline{{y}}$) for **{full_state}**,
        we use this sample mean as an estimate for $1/\\lambda$. This allows us to estimate $\\lambda$ as
        $\\hat{{\\lambda}} = 1 / \overline{{y}}$.
        For **{full_state}**, the average weekly ILI is **{ili_value.mean():.2f}%**, leading to an estimated
        $\\hat{{\\lambda}} \\approx {lambda_hat:.2f}$.
        If the red curve (the estimated Exponential PDF) visually matches the shape of the histogram well,
        it suggests that an Exponential distribution might be a reasonable, albeit simple, model to characterize
        the typical occurrence of ILI percentages.

        * **What does this $\hat{{\lambda}}$ for {full_state} tell us?**
        
            A smaller $\hat{{\lambda}}$ (which means a larger average ILI %, $1/\hat{{\lambda}}$) would suggest that,
            for **{full_state}**, higher ILI percentages are generally more common, or that the distribution of ILI % has
            a "heavier tail" (more spread towards higher values).
            Conversely, a larger $\hat{{\lambda}}$ (smaller average ILI %) would imply that ILI percentages for **{full_state}**
            tend to be lower more consistently.
        """
    )
else:
    # If there's no data or the mean ILI is zero, display a message instead of the distribution plot.
    st.write(f"Not enough data or zero mean ILI to plot distribution for {full_state}.")
