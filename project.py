import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

@st.cache_data
def load_data():
    df = pd.read_csv("ilidata.csv")
    df = df.dropna(subset=["ili"])
    df["weeks"] = df.groupby("state").cumcount()
    return df

df = load_data()

st.title("Influenza‑Like Illness (ILI) Trends, 2010 – 2025")
state_list = sorted(df["state"].unique())
state = st.selectbox("Choose a U.S. state", state_list)

state_df = df[df["state"] == state]

st.subheader(f"Weekly Percent ILI in {state}")
st.line_chart(state_df.set_index("weeks")["ili"])

st.subheader(f"Distribution of Weekly ILI % in {state}")
ili_vals = state_df["ili"].values
lambda_hat = 1.0 / ili_vals.mean()

fig, ax = plt.subplots()
ax.hist(ili_vals, bins=30, density=True, alpha=0.6, label="Observed ILI %")
x = np.linspace(0, ili_vals.max(), 300)
ax.plot(x, expon(scale=1/lambda_hat).pdf(x), lw=2, label=f"Exp(λ̂={lambda_hat:.2f})")
ax.set_xlabel("ILI (%)")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)

st.header("What does this mean?")
st.markdown(
    f"""
    **LLN & Exponential Fit Insight**  
    * For an exponential distribution, mean = 1/λ  
    * Sample mean → λ̂ = 1 / mean  
    * {state} average ILI: {ili_vals.mean():.2f}%, so λ̂ ≈ {lambda_hat:.2f}  
    * Histogram + PDF curve shows if exponential fits well  
    """
)
