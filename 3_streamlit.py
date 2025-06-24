import streamlit as st
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import softmax
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Load models
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    kw_model = KeyBERT()
    return tokenizer, model, kw_model

tokenizer, model, kw_model = load_models()

# Helper: Sentiment analysis
def get_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_token_type_ids=False  # <<< Prevents the crash
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = softmax(logits.numpy()[0])
    labels = ['negative', 'neutral', 'positive']
    return labels[scores.argmax()], scores

# Streamlit UI
st.title("YT Insight AI")
video_url = st.text_input("Paste YouTube video URL:", "")

if video_url:
    with st.spinner("Scraping comments..."):
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(video_url)
        comment_list = [c['text'] for c in comments][:200]

        if(len(comment_list)<1):
            st.success(f"No Comments Found for this Video!")

        else:
            st.success(f"Collected {len(comment_list)} comments!")

            # Sentiment Analysis
            with st.spinner("Analyzing sentiment..."):
                sentiments = [get_sentiment(c)[0] for c in comment_list]
                sentiment_counts = pd.Series(sentiments).value_counts()

                st.subheader("📊 Sentiment Distribution")
                st.bar_chart(sentiment_counts)

            # KeyBERT Keyword Extraction
            with st.spinner("Extracting keywords..."):
                text = " ".join(comment_list)
                keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=15)
                df_keywords = pd.DataFrame(keywords, columns=["Keyword", "Score"])

            from wordcloud import WordCloud

            st.subheader("☁️ Comments section Word Cloud - with KeyBert")

            # Convert keywords and scores into dictionary
            keyword_dict = {word: score for word, score in keywords}

            # Create word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_dict)

            # Display in Streamlit
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
