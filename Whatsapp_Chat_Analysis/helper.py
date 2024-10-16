from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
from transformers import pipeline

# Use your Hugging Face API Key
from huggingface_hub import InferenceClient

client = InferenceClient(api_key="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")

# Initialize URL extractor
extract = URLExtract()

def call_huggingface_llm(query, raw_chat_text):
    response = ""
    for message in client.chat_completion(
      model="mistralai/Mistral-Nemo-Instruct-2407",
      messages=[{"role": "user", "content": f"Answer the question : {query} according to the following chat: {raw_chat_text}"}],
      max_tokens=500,
      stream=True,
    ):
        response = response + message.choices[0].delta.content
    return response

def call_rag_model(query):
    summary = ""
    for message in client.chat_completion(
      model="mistralai/Mistral-Nemo-Instruct-2407",
      messages=[{"role": "user", "content": f"{query}"}],
      max_tokens=500,
      stream=True,
    ):
        summary = summary + message.choices[0].delta.content
    return summary

# Function to fetch statistics
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'user': 'name', 'count': 'percent'})
    return x, df

def create_wordcloud(selected_user, df):
    f = open('stop_hinglish', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    f = open('stop_hinglish', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

# Function to generate a summary using the LLM model
def generate_summary(raw_chat_text):
    summary = call_rag_model(query=f"Summarize this WhatsApp conversation: {raw_chat_text}")
    return summary

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

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

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap
