def generate_auto_insights(df, ai_df, profile_df, topic_df):
    insights = []

    filtered_df = df[df['user'] != 'group_notification']

    if not filtered_df.empty:
        most_active = filtered_df['user'].value_counts().idxmax()
        insights.append(f"{most_active} is the most active user in the chat.")

        avg_msg_len = filtered_df.groupby('user')['message'].apply(
            lambda x: sum(len(str(i)) for i in x) / max(len(x), 1)
        )
        if not avg_msg_len.empty:
            detailed_user = avg_msg_len.idxmax()
            insights.append(f"{detailed_user} tends to send the longest messages on average.")

    if not ai_df.empty and 'sentiment' in ai_df.columns:
        pos_count = ai_df[ai_df['sentiment'] == 'POSITIVE']['user'].value_counts()
        neg_count = ai_df[ai_df['sentiment'] == 'NEGATIVE']['user'].value_counts()

        if not pos_count.empty:
            insights.append(f"{pos_count.idxmax()} has the highest number of positive-toned messages.")

        if not neg_count.empty:
            insights.append(f"{neg_count.idxmax()} has the highest number of negative-toned messages.")

    if not ai_df.empty and 'emotion' in ai_df.columns:
        top_emotion = ai_df['emotion'].value_counts()
        if not top_emotion.empty:
            insights.append(f"The dominant emotion across the chat is {top_emotion.idxmax()}.")

    if not profile_df.empty and 'night_activity_ratio' in profile_df.columns:
        night_user = profile_df.sort_values('night_activity_ratio', ascending=False).iloc[0]['user']
        insights.append(f"{night_user} is the strongest late-night contributor.")

    if not topic_df.empty and 'topic_id' in topic_df.columns:
        dominant_topic = topic_df['topic_id'].value_counts().idxmax()
        insights.append(f"Topic {dominant_topic} appears most frequently in the conversation.")

    return insights