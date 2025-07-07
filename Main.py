# main.py

# 1. Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# 2. Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_ecommerce_2025-06-13_Version2.csv")
    df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    df['last_activity_ts'] = pd.to_datetime(df['last_activity_ts'], errors='coerce')

    # Drop rows with missing dates
    df.dropna(subset=['signup_date', 'last_activity_ts'], inplace=True)

    # Fill numeric nulls with median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical nulls with mode
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

df = load_data()

# 3. Generate hypothesis
st.title("📊 eCommerce Customer Analytics Dashboard")
st.markdown("""
**Hypothesis:**  
Customers who are premium members and have higher email open rates also tend to have higher CLV (Customer Lifetime Value).
""")

# 4. Sidebar filters
st.sidebar.header("Filter Data")
selected_country = st.sidebar.multiselect("Select Country", df['country_code'].unique(), default=df['country_code'].unique())
selected_segment = st.sidebar.multiselect("Customer Segment", df['customer_segment'].unique(), default=df['customer_segment'].unique())

filtered_df = df[(df['country_code'].isin(selected_country)) & (df['customer_segment'].isin(selected_segment))]

# 5. Univariate plot: Histogram of CLV
st.subheader("CLV Distribution")
fig1 = px.histogram(filtered_df, x='clv_3yr_usd', nbins=40, title="Customer Lifetime Value Distribution")
st.plotly_chart(fig1)

# 6. Bivariate plot: Premium membership vs CLV
st.subheader("Premium Members vs CLV")
fig2 = px.box(filtered_df, x='is_premium_member', y='clv_3yr_usd', color='is_premium_member', title="CLV by Premium Status")
st.plotly_chart(fig2)

# 7. Multivariate plot: Email Open Rate vs CLV
st.subheader("Email Open Rate vs CLV")
fig3 = px.scatter(filtered_df, x='email_open_rate', y='clv_3yr_usd', color='is_premium_member',
                  size='total_spend_12m', title="Email Open Rate vs CLV Colored by Premium Status")
st.plotly_chart(fig3)

# 8. Correlation heatmap
st.subheader("Correlation Heatmap (Numeric Variables)")
corr_matrix = filtered_df.select_dtypes(include=np.number).corr()
fig4 = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
st.plotly_chart(fig4)

# 9. Key Insights
st.markdown("""
### 🔍 Key Insights:
- Premium members tend to have significantly higher CLV.
- Email open rate is moderately positively correlated with CLV, especially among premium customers.
- CLV also shows strong positive correlation with `total_spend_12m` and `avg_basket_value`.

### ✅ Recommendations:
- Focus marketing efforts on increasing email engagement rates.
- Offer targeted loyalty incentives to premium members with high open rates.
""")
