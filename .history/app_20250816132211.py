import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from googleapiclient.discovery import build
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# -------------------------
# Load Naive Bayes Model & Vectorizer
# -------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model_nb.pkl")
    vectorizer = joblib.load("vectorizer_nb.pkl")
    return model, vectorizer


model, vectorizer = load_model()

# -------------------------
# YouTube API Setup
# -------------------------
api_key = st.secrets["api"]["youtube_api_key"]


def get_youtube_comments(video_id, max_results=50):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText"
    )
    response = request.execute()

    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    return comments


# -------------------------
# Sentiment Prediction
# -------------------------
def predict_sentiment(comments):
    X = vectorizer.transform(comments)
    predictions = model.predict(X)
    return predictions


# -------------------------
# Streamlit App
# -------------------------
st.title("🎯 Naïve Bayes YouTube Sentiment Analysis")
st.write("This app analyzes YouTube video comments and predicts their sentiment.")

# 1️⃣ Enter YouTube Video URL
video_url = st.text_input("Enter YouTube Video URL:")

if st.button("Fetch Comments"):
    if video_url:
        # Extract video ID correctly from both long and short YouTube links
        if "v=" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[-1].split("?")[0]
        else:
            st.error("Invalid YouTube URL format.")
            st.stop()

        try:
            comments = get_youtube_comments(video_id)
            if comments:
                df = pd.DataFrame({"comment": comments})
                st.session_state.df = df
                st.success(f"Fetched {len(comments)} comments successfully!")
            else:
                st.warning("No comments found for this video.")
        except Exception as e:
            st.error(f"Error fetching comments: {e}")
    else:
        st.error("Please enter a YouTube URL.")

# 2️⃣ Upload CSV Option
uploaded_file = st.file_uploader(
    "Or upload a CSV file with a 'comment' column", type=["csv"]
)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.success(f"Uploaded file with {len(df)} comments.")

# 3️⃣ Run Sentiment Analysis
if "df" in st.session_state and st.button("Run Sentiment Analysis"):
    df = st.session_state.df
    predictions = predict_sentiment(df["comment"])
    df["sentiment"] = predictions

    st.write("### Sentiment Results")
    st.dataframe(df)

    # Pie Chart
    st.write("### Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    st.pyplot(fig)

    # Word Cloud
    st.write("### Word Cloud of Comments")
    all_text = " ".join(df["comment"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        all_text
    )
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Download Results
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results as CSV", csv, "sentiment_results.csv", "text/csv"
    )
