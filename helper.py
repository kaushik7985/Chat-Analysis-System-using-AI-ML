from collections import Counter
import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji

extract = URLExtract()


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    media_messages = 0
    links = []

    for message in df['message']:
        words.extend(str(message).split())
        links.extend(extract.find_urls(str(message)))
        if str(message).strip() == '<Media omitted>':
            media_messages += 1

    return num_messages, len(words), media_messages, len(links)


def most_busy_users(df):
    x = df[df['user'] != 'group_notification']['user'].value_counts().head()
    df_percent = round(
        (df[df['user'] != 'group_notification']['user'].value_counts() / df[df['user'] != 'group_notification'].shape[0]) * 100,
        2
    ).reset_index()
    df_percent.columns = ['name', 'percent']
    return x, df_percent


def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    text = ' '.join(temp['message'].astype(str).tolist())

    if not text.strip():
        text = "No meaningful text found"

    wc = WordCloud(width=800, height=400, min_font_size=10, background_color='white')
    return wc.generate(text)


def most_common_words(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']

    words = []
    for message in temp['message']:
        for word in str(message).lower().split():
            if word not in ['<media', 'omitted>']:
                words.append(word)

    common_df = pd.DataFrame(Counter(words).most_common(20))
    if not common_df.empty:
        common_df.columns = ['word', 'count']
    else:
        common_df = pd.DataFrame(columns=['word', 'count'])

    return common_df


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        for c in str(message):
            if c in emoji.EMOJI_DATA:
                emojis.append(c)

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    if not emoji_df.empty:
        emoji_df.columns = ['emoji', 'count']
    else:
        emoji_df = pd.DataFrame(columns=['emoji', 'count'])

    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline[['time', 'message']]


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily = df.groupby('date').count()['message'].reset_index()
    return daily


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(
        index='day_name',
        columns='period',
        values='message',
        aggfunc='count'
    ).fillna(0)

    return user_heatmap