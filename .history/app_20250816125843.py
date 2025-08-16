import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re

# ---------------------------
# Load API Key securely
# ---------------------------
api_key = st.secrets["api"]["youtube_api_key"]

# ---------------------------
# Function to fetch YouTube comments
# ---------------------------
def get_youtube_comments(video_id, max_results=50):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    response = request.execute()
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
    return comments

# ---------------------------
# Simple preprocessing
# ---------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower().strip()

# ---------------------------
# Train simple NB pipeline
# ---------------------------
def train_naive_bayes():
    train_data = pd.DataFrame({
        "comment": [
            "I love this video!", "Great job!", "Amazing content",
            "This is bad", "Terrible experience", "Worst ever"
        ],
        "label": ["positive", "positive", "positive", "negative", "negative", "negative"]
    })
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(train_data["comment"], train_data["label"])
    return pipeline

# ---------------------------
# UI
# ---------------------------
st.title("YouTube Comment Sentiment Analysis (Naïve Bayes Model)")

choice = st.radio("Select Input Method", ["Fetch from YouTube", "Upload CSV"])

if choice == "Fetch from YouTube":
    video_url = st.text_input("Enter YouTube Video URL:  if st.button("Fetch Comments"):
    if video_url:
        video_id = extract_video_id(video_url)  # ✅ New code
        if video_id:
            comments = get_youtube_comments(video_id)
            if comments:
                df = pd.DataFrame({"comment": comments})
                st.session_state.df = df
            else:
                st.warning("No comments found for this video.")
        else:
            st.error("Invalid YouTube URL format.")

            if comments:
                df = pd.DataFrame({"comment": comments})
                st.session_state.df = df
                st.success("Comments fetched successfully!")
            else:
                st.error("No comments found.")
elif choice == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "comment" in df.columns:
            st.session_state.df = df
            st.success("CSV loaded successfully!")
        else:
            st.error("CSV must have a 'comment' column.")

# ---------------------------
# Run Naïve Bayes
# ---------------------------
if "df" in st.session_state:
    if st.button("Run Naïve Bayes Model"):
        model = train_naive_bayes()
        st.session_state.df["clean_comment"] = st.session_state.df["comment"].apply(clean_text)
        st.session_state.df["sentiment"] = model.predict(st.session_state.df["clean_comment"])
        
        st.subheader("Results")
        st.dataframe(st.session_state.df)

        # Pie Chart
        sentiment_counts = st.session_state.df["sentiment"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        # Word Cloud
        text = " ".join(st.session_state.df["clean_comment"])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)