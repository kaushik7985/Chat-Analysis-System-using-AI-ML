import re
import pandas as pd
import emoji
from urlextract import URLExtract
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline

extractor = URLExtract()

# -----------------------------
# MODEL LOADING
# -----------------------------
# Sentiment
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Emotion
emotion_pipe = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# Summary
# summarization task supported hone par ye chalega
# agar future me issue aaye to text2text-generation fallback use karenge
summarizer_pipe = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)


# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text_for_ai(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# SENTIMENT + EMOTION
# -----------------------------
def add_sentiment_emotion(df):
    temp = df.copy()
    temp = temp[temp["user"] != "group_notification"]
    temp = temp[temp["message"].astype(str).str.strip() != ""]

    if temp.empty:
        temp["sentiment"] = []
        temp["emotion"] = []
        return temp

    temp["clean_message"] = temp["message"].apply(clean_text_for_ai)

    # IMPORTANT FIX:
    # Transformer models usually have token limit ~512
    # Isliye long message ko short kar rahe hain
    temp["clean_message"] = temp["clean_message"].apply(lambda x: x[:300])

    texts = temp["clean_message"].tolist()

    sentiments = []
    emotions = []

    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            sent_out = sentiment_pipe(batch, truncation=True)
            emo_out = emotion_pipe(batch, truncation=True)
        except Exception:
            continue

        for s in sent_out:
            sentiments.append(s["label"])

        for e in emo_out:
            best_emotion = max(e, key=lambda x: x["score"])["label"]
            emotions.append(best_emotion)

    # Safety: agar kisi batch me issue hua ho
    valid_len = min(len(temp), len(sentiments), len(emotions))
    temp = temp.iloc[:valid_len].copy()
    temp["sentiment"] = sentiments[:valid_len]
    temp["emotion"] = emotions[:valid_len]

    return temp


# -----------------------------
# TOPIC MODELING
# -----------------------------
def get_topic_model(df, n_topics=5):
    temp = df.copy()
    temp = temp[temp["user"] != "group_notification"]
    temp = temp[temp["message"].astype(str).str.len() > 3]

    docs = temp["message"].apply(clean_text_for_ai).tolist()

    if len(docs) < 5:
        return None, None, None, temp

    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=2
    )

    try:
        doc_term_matrix = vectorizer.fit_transform(docs)
    except Exception:
        return None, None, None, temp

    if doc_term_matrix.shape[1] == 0:
        return None, None, None, temp

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )
    lda.fit(doc_term_matrix)

    return lda, vectorizer, doc_term_matrix, temp


def extract_topics(lda, vectorizer, n_words=8):
    if lda is None or vectorizer is None:
        return {}

    words = vectorizer.get_feature_names_out()
    topics = {}

    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-n_words:][::-1]]
        topics[f"Topic {idx + 1}"] = top_words

    return topics


def assign_topics_to_messages(lda, doc_term_matrix, temp_df):
    if lda is None or doc_term_matrix is None or temp_df.empty:
        return pd.DataFrame()

    topic_values = lda.transform(doc_term_matrix)
    assigned_topics = topic_values.argmax(axis=1)

    result = temp_df.copy()
    result["topic_id"] = assigned_topics + 1
    return result


# -----------------------------
# USER PROFILING
# -----------------------------
def count_emojis(text):
    return sum(1 for char in str(text) if char in emoji.EMOJI_DATA)


def build_user_profiles(df):
    temp = df.copy()
    temp = temp[temp["user"] != "group_notification"]

    if temp.empty:
        return pd.DataFrame(columns=[
            "user",
            "total_messages",
            "avg_message_length",
            "avg_words_per_message",
            "emoji_count",
            "link_count",
            "night_activity_ratio",
            "most_active_hour",
            "profile_tag"
        ])

    temp["msg_length"] = temp["message"].apply(lambda x: len(str(x)))
    temp["word_count"] = temp["message"].apply(lambda x: len(str(x).split()))
    temp["emoji_count"] = temp["message"].apply(count_emojis)
    temp["link_count"] = temp["message"].apply(lambda x: len(extractor.find_urls(str(x))))
    temp["is_night"] = temp["hour"].apply(lambda x: 1 if x >= 23 or x <= 5 else 0)

    profile = temp.groupby("user").agg({
        "message": "count",
        "msg_length": "mean",
        "word_count": "mean",
        "emoji_count": "sum",
        "link_count": "sum",
        "is_night": "mean"
    }).reset_index()

    profile.rename(columns={
        "message": "total_messages",
        "msg_length": "avg_message_length",
        "word_count": "avg_words_per_message",
        "is_night": "night_activity_ratio"
    }, inplace=True)

    active_hour = temp.groupby(["user", "hour"]).size().reset_index(name="count")
    idx = active_hour.groupby("user")["count"].idxmax()
    best_hours = active_hour.loc[idx][["user", "hour"]]
    best_hours.rename(columns={"hour": "most_active_hour"}, inplace=True)

    profile = profile.merge(best_hours, on="user", how="left")

    labels = []
    median_emoji = profile["emoji_count"].median() if not profile.empty else 0

    for _, row in profile.iterrows():
        tag = []

        if row["avg_message_length"] > 40:
            tag.append("Detailed Sender")
        else:
            tag.append("Short Replier")

        if row["emoji_count"] > median_emoji:
            tag.append("Emoji Friendly")

        if row["night_activity_ratio"] > 0.30:
            tag.append("Night Owl")

        labels.append(", ".join(tag))

    profile["profile_tag"] = labels

    return profile


# -----------------------------
# SUMMARY
# -----------------------------
def generate_chat_summary(df, max_chunk_chars=1500):
    temp = df.copy()
    temp = temp[temp["user"] != "group_notification"]

    all_text = " ".join(temp["message"].astype(str).tolist())
    all_text = clean_text_for_ai(all_text)

    if not all_text.strip():
        return "No summary could be generated."

    chunks = []
    start = 0
    while start < len(all_text):
        chunk = all_text[start:start + max_chunk_chars]
        chunks.append(chunk)
        start += max_chunk_chars

    partial_summaries = []

    for chunk in chunks[:5]:
        if len(chunk.split()) < 20:
            continue

        try:
            summary = summarizer_pipe(
                chunk,
                max_length=80,
                min_length=20,
                do_sample=False,
                truncation=True
            )
            partial_summaries.append(summary[0]["summary_text"])
        except Exception:
            continue

    if not partial_summaries:
        return "Summary generation failed for this chat."

    merged = " ".join(partial_summaries)

    try:
        final_summary = summarizer_pipe(
            merged,
            max_length=100,
            min_length=25,
            do_sample=False,
            truncation=True
        )
        return final_summary[0]["summary_text"]
    except Exception:
        return merged


# -----------------------------
# DISTRIBUTION TABLES
# -----------------------------
def user_sentiment_distribution(ai_df):
    if ai_df.empty:
        return pd.DataFrame()
    return pd.crosstab(ai_df["user"], ai_df["sentiment"])


def user_emotion_distribution(ai_df):
    if ai_df.empty:
        return pd.DataFrame()
    return pd.crosstab(ai_df["user"], ai_df["emotion"])