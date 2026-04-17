import re
import pandas as pd


def preprocess(data):
    # Supports common WhatsApp export format like:
    # 12/08/23, 9:14 pm - User: Message
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?[apAP][mM])\s-\s'
    parts = re.split(pattern, data)

    if len(parts) < 3:
        return pd.DataFrame(columns=[
            'message_date', 'user', 'message', 'year', 'month', 'day',
            'hour', 'minute', 'day_name', 'date'
        ])

    dates = parts[1::2]
    messages = parts[2::2]

    df = pd.DataFrame({
        'message_date': dates,
        'user_message': messages
    })

    df['message_date'] = pd.to_datetime(
        df['message_date'],
        format='%d/%m/%y, %I:%M %p',
        errors='coerce'
    )

    # fallback parser
    mask = df['message_date'].isna()
    if mask.any():
        df.loc[mask, 'message_date'] = pd.to_datetime(
            df.loc[mask, 'message_date'],
            errors='coerce'
        )

    df.dropna(subset=['message_date'], inplace=True)

    users = []
    messages_clean = []

    for msg in df['user_message']:
        entry = re.split(r'([^:]+):\s', msg, maxsplit=1)

        if len(entry) >= 3:
            users.append(entry[1].strip())
            messages_clean.append(entry[2].strip())
        else:
            users.append('group_notification')
            messages_clean.append(str(msg).strip())

    df['user'] = users
    df['message'] = messages_clean

    df['year'] = df['message_date'].dt.year
    df['month'] = df['message_date'].dt.month_name()
    df['day'] = df['message_date'].dt.day
    df['hour'] = df['message_date'].dt.hour
    df['minute'] = df['message_date'].dt.minute
    df['day_name'] = df['message_date'].dt.day_name()
    df['date'] = df['message_date'].dt.date

    period = []
    for hour in df['hour']:
        start = hour
        end = (hour + 1) % 24
        period.append(f"{start:02d}-{end:02d}")
    df['period'] = period

    df.drop(columns=['user_message'], inplace=True)

    return df