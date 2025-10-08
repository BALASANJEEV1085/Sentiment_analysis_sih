from collections import Counter
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import base64

def bar_sentiment_counts(df, sentiment_col='sentiment'):
    counts = df[sentiment_col].value_counts().reset_index()
    counts.columns = ['sentiment','count']
    chart = alt.Chart(counts).mark_bar().encode(
        x=alt.X('sentiment:N', sort=['positive','neutral','negative']),
        y='count:Q',
        color='sentiment:N'
    ).properties(width=600, height=300, title='Sentiment Distribution')
    return chart

def trend_over_time(df, date_col, sentiment_col='sentiment', freq='D'):
    df2 = df.copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
    df2 = df2.dropna(subset=[date_col])
    df2['period'] = df2[date_col].dt.to_period(freq).dt.to_timestamp()
    grouped = df2.groupby(['period', sentiment_col]).size().reset_index(name='count')
    chart = alt.Chart(grouped).mark_line(point=True).encode(
        x='period:T',
        y='count:Q',
        color=alt.Color(sentiment_col + ':N'),
        tooltip=['period:T','count:Q', sentiment_col+':N']
    ).properties(width=700, height=300, title='Sentiment Trends Over Time')
    return chart

def top_words_bar(df, clean_col='clean_text', top_n=20):
    texts = df[clean_col].astype(str).tolist()
    words = Counter()
    for t in texts:
        words.update(t.split())
    most = words.most_common(top_n)
    dfm = pd.DataFrame(most, columns=['word','count'])
    chart = alt.Chart(dfm).mark_bar().encode(
        x='count:Q',
        y=alt.Y('word:N', sort='-x'),
        tooltip=['word','count']
    ).properties(width=600, height=400, title='Most Common Words')
    return chart

def generate_wordcloud_image(df, clean_col='clean_text', max_words=150):
    text = " ".join(df[clean_col].astype(str).tolist())
    wc = WordCloud(width=800, height=400, background_color='white', max_words=max_words).generate(text)
    # return as bytes image
    img_buf = BytesIO()
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close()
    img_buf.seek(0)
    return img_buf
