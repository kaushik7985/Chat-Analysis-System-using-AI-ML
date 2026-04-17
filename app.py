import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import preprocessor
import helper

from ai_features import (
    add_sentiment_emotion,
    get_topic_model,
    extract_topics,
    assign_topics_to_messages,
    build_user_profiles,
    generate_chat_summary,
    user_sentiment_distribution,
    user_emotion_distribution
)

from insights import generate_auto_insights

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Intelligent Chat Analysis System using AI/ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    .hero {
        padding: 28px;
        border-radius: 22px;
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
    }

    .hero p {
        margin-top: 10px;
        font-size: 1rem;
        opacity: 0.95;
    }

    .badge {
        display: inline-block;
        margin-top: 14px;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.18);
        font-size: 0.9rem;
        font-weight: 600;
    }

    .section-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 20px;
    }

    .metric-card {
        background: linear-gradient(135deg, #eef2ff, #ecfeff);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 18px 14px;
        text-align: center;
        margin-bottom: 10px;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #334155;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
    }

    .small-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 12px;
        color: #0f172a;
    }

    .topic-box {
        background: #f1f5f9;
        border-left: 5px solid #06b6d4;
        padding: 14px 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        color: #0f172a;
    }

    .insight-box {
        background: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 14px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        color: #065f46;
        font-weight: 500;
    }

    .summary-box {
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 18px;
        border-radius: 14px;
        color: #1e3a8a;
        line-height: 1.7;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🤖 AI Chat Dashboard")
st.sidebar.markdown("**Intelligent Chat Analysis System using AI/ML**")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("📂 Choose a chat file", type=["txt"])

st.markdown("""
<div class="hero">
    <h1>Intelligent Chat Analysis System using AI/ML</h1>
    <p>
        Analyze, understand, and summarize conversations using
        Artificial Intelligence, Natural Language Processing, and Machine Learning.
    </p>
    <div class="badge">🚀 AI-Powered Conversation Intelligence Dashboard</div>
</div>
""", unsafe_allow_html=True)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8", errors="ignore")

    df = preprocessor.preprocess(data)

    if df.empty:
        st.error("Could not parse chat file. Please export chat again and try.")
        st.stop()

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("👤 Show analysis for", user_list)

    if st.sidebar.button("✨ Show Analysis", width="stretch"):

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">📊 Basic Statistics</div>', unsafe_allow_html=True)

        num_messages, words, media_messages, links = helper.fetch_stats(selected_user, df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Messages</div>
                <div class="metric-value">{num_messages}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Words</div>
                <div class="metric-value">{words}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Media Shared</div>
                <div class="metric-value">{media_messages}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Links Shared</div>
                <div class="metric-value">{links}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if selected_user == 'Overall':
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="small-title">🔥 Most Busy Users</div>', unsafe_allow_html=True)

            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns([2, 1])

            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(x.index, x.values)
                plt.xticks(rotation=30)
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df, width="stretch")

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">📈 Activity Timeline</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Monthly Timeline**")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(timeline['time'], timeline['message'], marker='o')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("**Daily Timeline**")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(daily_timeline['date'], daily_timeline['message'])
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">🗓️ Activity Map</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Most Busy Day**")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("**Most Busy Month**")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(busy_month.index, busy_month.values)
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("**Weekly Activity Heatmap**")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(user_heatmap, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">☁️ Text Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**WordCloud**")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(df_wc)
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("**Most Common Words**")
            most_common_df = helper.most_common_words(selected_user, df)
            if not most_common_df.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(most_common_df['word'], most_common_df['count'])
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No common words found.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">😄 Emoji Analysis</div>', unsafe_allow_html=True)

        emoji_df = helper.emoji_helper(selected_user, df)

        if not emoji_df.empty:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.dataframe(emoji_df, width="stretch")

            with col2:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(
                    emoji_df['count'].head(),
                    labels=emoji_df['emoji'].head(),
                    autopct="%0.2f"
                )
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("No emojis found.")

        st.markdown('</div>', unsafe_allow_html=True)

        if selected_user != "Overall":
            df_ai_base = df[df['user'] == selected_user].copy()
        else:
            df_ai_base = df.copy()

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">🧠 Sentiment and Emotion Analysis</div>', unsafe_allow_html=True)

        with st.spinner("Running sentiment and emotion analysis..."):
            ai_df = add_sentiment_emotion(df_ai_base)

        if not ai_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Sentiment Distribution**")
                st.bar_chart(ai_df['sentiment'].value_counts())

            with col2:
                st.markdown("**Emotion Distribution**")
                st.bar_chart(ai_df['emotion'].value_counts())

            st.markdown("**Per-user Sentiment Distribution**")
            sent_dist = user_sentiment_distribution(ai_df)
            if not sent_dist.empty:
                st.dataframe(sent_dist, width="stretch")

            st.markdown("**Per-user Emotion Distribution**")
            emo_dist = user_emotion_distribution(ai_df)
            if not emo_dist.empty:
                st.dataframe(emo_dist, width="stretch")
        else:
            st.info("No valid messages available for sentiment/emotion analysis.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">🧩 Topic Extraction</div>', unsafe_allow_html=True)

        lda, vectorizer, dtm, topic_temp_df = get_topic_model(df_ai_base, n_topics=5)
        topics = extract_topics(lda, vectorizer)
        topic_df = assign_topics_to_messages(lda, dtm, topic_temp_df)

        if topics:
            for topic, words in topics.items():
                st.markdown(f"""
                <div class="topic-box">
                    <b>{topic}</b><br>
                    {", ".join(words)}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Not enough text data for topic extraction.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">👤 User Behavior Profiling</div>', unsafe_allow_html=True)

        profile_df = build_user_profiles(df_ai_base)
        if not profile_df.empty:
            st.dataframe(profile_df, width="stretch")
        else:
            st.info("Not enough data for user profiling.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">📝 Chat Summary</div>', unsafe_allow_html=True)

        with st.spinner("Generating summary..."):
            summary = generate_chat_summary(df_ai_base)

        st.markdown(f"""
        <div class="summary-box">
            {summary}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-title">💡 Auto Insights</div>', unsafe_allow_html=True)

        insights = generate_auto_insights(df_ai_base, ai_df, profile_df, topic_df)

        if insights:
            for ins in insights:
                st.markdown(f"""
                <div class="insight-box">
                    {ins}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No insights could be generated.")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="section-card">
        <div class="small-title">Welcome</div>
        Upload your exported strchat <b>.txt</b> file from the sidebar to begin analysis.
    </div>
    """, unsafe_allow_html=True)