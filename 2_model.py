from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from youtube_comment_downloader import YoutubeCommentDownloader
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForSequenceClassification.from_pretrained(MODEL)


downloader = YoutubeCommentDownloader()
comments = downloader.get_comments_from_url('https://www.youtube.com/watch?v=oLV6V70SKro')

# Store comments in a list
comment_list = []
for i in comments:
    comment_list.append(i['text'])  # Or comment['textOriginal']


def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = softmax(logits.numpy()[0])
    labels = ['negative', 'neutral', 'positive']
    return labels[scores.argmax()], scores


## Sentiment of Comments

# sent_set = set()

# for i, comment in enumerate(comment_list):
#     sentiment, scores = get_sentiment(comment)
#     # print(f"{i+1}. [{sentiment}] {comment}")
#     sent_set.add(sentiment)

# print(sent_set)

# vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=15)
# X = vectorizer.fit_transform(comment_list)
# word_freq = X.sum(axis=0).A1
# words = vectorizer.get_feature_names_out()

# Plot
# df_freq = pd.DataFrame({'word': words, 'freq': word_freq})
# df_freq.sort_values('freq', ascending=False).plot.bar(x='word', y='freq', legend=False, figsize=(10,5), color='skyblue')
# plt.title("Top Things People Are Talking About 🎯")
# plt.xlabel("Topic/Word")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()


from keybert import KeyBERT

kw_model = KeyBERT()
text = " ".join(comment_list)
keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=15)

# Convert for plotting
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(keywords, columns=["Keyword", "Score"])
df.plot.barh(x='Keyword', y='Score', color='orange', figsize=(8,6))
plt.title("💡 Smart Insights: What People Actually Care About")
plt.gca().invert_yaxis()
plt.show()