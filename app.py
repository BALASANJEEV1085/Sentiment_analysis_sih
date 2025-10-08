import streamlit as st
import pandas as pd
import numpy as np
import os
from src.data_loader import load_uploaded_file
from src.preprocessor import preprocess_dataframe
from src.trainer import vader_sentiment_label, vader_sentiment_score
from src.visuals import bar_sentiment_counts, top_words_bar, generate_wordcloud_image
import matplotlib.pyplot as plt

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(layout="wide", page_title="Flexible Sentiment Analyzer")

st.title("📊 Flexible Sentiment Analysis Dashboard (CSV / Excel / JSON)")

# -----------------------------
# Sidebar: Upload & Settings
# -----------------------------
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel / JSON", type=['csv', 'xlsx', 'xls', 'json'])

# -----------------------------
# If File Uploaded
# -----------------------------
if uploaded_file:
    try:
        df = load_uploaded_file(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load file: {e}")
        st.stop()

    st.write("### 📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Detect text-like columns
    text_candidates = [c for c in df.columns if df[c].dtype == object or df[c].dtype.name.startswith('string')]
    if not text_candidates:
        text_candidates = list(df.columns)

    # Simplified Sidebar (only text column + run button)
    st.sidebar.subheader("Select Text Column")
    text_column = st.sidebar.selectbox("Text column (required)", text_candidates, index=0)

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Run Analysis ▶")

    # -----------------------------
    # Run Analysis
    # -----------------------------
    df = preprocess_dataframe(df, text_column)

    if run_button:
        with st.spinner("Analyzing sentiments..."):
            # Always use VADER
            df['vader_scores'] = df['clean_text'].apply(lambda t: vader_sentiment_score(t))
            df['compound'] = df['vader_scores'].apply(lambda d: d['compound'])
            df['sentiment'] = df['compound'].apply(
                lambda c: 'positive' if c >= 0.05 else ('negative' if c <= -0.05 else 'neutral')
            )

            # -----------------------------
            # Summary Metrics
            # -----------------------------
            total = len(df)
            counts = df['sentiment'].value_counts().to_dict()
            positive = counts.get('positive', 0)
            neutral = counts.get('neutral', 0)
            negative = counts.get('negative', 0)

            st.subheader("📈 Summary Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", total)
            col2.metric("Positive", f"{positive}", f"{positive / total:.1%}" if total else "")
            col3.metric("Neutral", f"{neutral}", f"{neutral / total:.1%}" if total else "")
            col4.metric("Negative", f"{negative}", f"{negative / total:.1%}" if total else "")

            # -----------------------------
            # Charts & Visuals
            # -----------------------------
            st.markdown("---")
            st.subheader("📊 Charts & Visuals")

            c1, c2 = st.columns([1, 1])
            with c1:
                st.altair_chart(bar_sentiment_counts(df, 'sentiment'), use_container_width=True)
            with c2:
                img_buf = generate_wordcloud_image(df, clean_col='clean_text')
                st.image(img_buf, caption="Word Cloud", use_column_width=True)

            # -----------------------------
            # Most Common Words
            # -----------------------------
            st.markdown("---")
            st.subheader("🗣️ Most Common Words")
            st.altair_chart(top_words_bar(df, clean_col='clean_text', top_n=30), use_container_width=True)

            # -----------------------------
            # Sample Results
            # -----------------------------
            st.markdown("---")
            st.subheader("🧾 Sample Results")
            st.dataframe(df[[text_column, 'clean_text', 'sentiment']].head(200))

            # -----------------------------
            # Download Results
            # -----------------------------
            st.markdown("---")
            st.subheader("💾 Download Results")

            @st.cache_data
            def convert_df_to_csv(dff):
                return dff.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(df)
            st.download_button(
                "Download analysis CSV",
                csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )

else:
    st.info("📥 Upload a dataset (CSV, Excel, or JSON) from the sidebar to start.")
    st.write("""
    **Usage Notes**
    - Upload any text dataset (CSV / Excel / JSON)
    - Choose the column containing text
    - Click **Run Analysis ▶** to start
    - The app uses **VADER** (rule-based) sentiment analysis
    - Works best for English text
    """)

