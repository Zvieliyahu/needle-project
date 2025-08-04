import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# create graph of topics vs amount of times
def graph_count_topics(df):
    # Split the commas
    df_expanded = df.assign(topic=df['topics'].str.split(',')).explode('topic')
    df_expanded['topic'] = df_expanded['topic'].str.strip()
    topic_counts = df_expanded['topic'].value_counts()

    # Plot bars at numeric positions
    plt.figure(figsize=(14, 7))
    positions = range(len(topic_counts))
    plt.bar(positions, topic_counts.values)
    plt.xticks(positions, topic_counts.index, rotation=45, ha='right')
    plt.title('Number of Times Each Topic Appears')
    plt.xlabel('Topic')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("TopicsVsCount.png")

# create graph of topics vs party
def graph_party_topic(df):
    # Explode topics (split comma, strip)
    df_expanded = df.assign(topic=df['topics'].str.split(',')).explode('topic')
    df_expanded['topic'] = df_expanded['topic'].str.strip()

    # Group by party and topic, count occurrences
    counts = df_expanded.groupby(['Party', 'topic']).size().reset_index(name='count')

    # Pivot: topics as index, parties as columns
    pivot_df = counts.pivot(index='topic', columns='Party', values='count').fillna(0)

    # Sort topics by total count descending (optional)
    pivot_df['total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('total', ascending=False).drop(columns='total')

    # Plot grouped bar chart
    pivot_df.plot(kind='bar', figsize=(16, 8))

    plt.title('Number of Times Each Topic Appears per Party')
    plt.xlabel('Topic')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Party')
    plt.tight_layout()
    plt.savefig("TopicCountsByAllParty.png")
    df_expanded = df_expanded[df_expanded['Party'].isin(['Republican', 'Democratic'])]

    # Group by party and topic, count occurrences
    counts = df_expanded.groupby(['Party', 'topic']).size().reset_index(name='count')

    # Pivot to have topics as index and parties as columns
    pivot_df = counts.pivot(index='topic', columns='Party', values='count').fillna(0)

    # Optional: sort topics by total count descending
    pivot_df['total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('total', ascending=False).drop(columns='total')

    # Plot grouped bar chart
    pivot_df.plot(kind='bar', figsize=(16,8))

    plt.title('Number of Times Each Topic Appears for Republicans and Democrats')
    plt.xlabel('Topic')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Party')
    plt.tight_layout()
    plt.savefig("TopicCountsByOnlyRandDPartys.png")

# create graph of topics vs time
def graph_time_topics(df):
    plt.clf()
    df['year'] = pd.to_datetime(df['date']).dt.year

    df_expanded = df.assign(topic=df['topics'].str.split(',')).explode('topic')
    df_expanded['topic'] = df_expanded['topic'].str.strip()

    min_year = df_expanded['year'].min()
    start_decade = (min_year // 10) * 10

    max_year = df_expanded['year'].max()
    end_decade = ((max_year // 10) + 1) * 10 - 1  # e.g. 2020 → 2029

    # Create decade bins edges: e.g. 1900, 1910, 1920, ..., 2030 (one edge beyond last bin)
    bins = np.arange(start_decade, end_decade + 2, 10)  # +2 to ensure last edge includes 2029
    labels = [f"{b}-{b + 9}" for b in bins[:-1]]

    df_expanded['decade_bin'] = pd.cut(
        df_expanded['year'], bins=bins, labels=labels, right=True, include_lowest=True
    )

    # Group and pivot like before
    topic_trends = df_expanded.groupby(['decade_bin', 'topic']).size().reset_index(name='count')
    pivot_trends = topic_trends.pivot(index='decade_bin', columns='topic', values='count').fillna(0)

    top_topics = pivot_trends.sum().sort_values(ascending=False).head(10).index
    pivot_trends_top = pivot_trends[top_topics]

    fig, ax = plt.subplots(figsize=(24, 8))
    pivot_trends_top.plot(ax=ax, linewidth=2)

    # Set x ticks to be at each decade start (numeric positions)
    decade_starts = bins[:-1]  # e.g. [1900, 1910, ..., 2020]
    xticks_pos = range(len(decade_starts))
    xticks_labels = [str(d) for d in decade_starts]

    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_labels, rotation=45, ha='right')

    ax.set_title('Topic Trends by Decade (Top 10 Topics)')
    ax.set_xlabel('Decade Start Year')
    ax.set_ylabel('Count')
    ax.legend(title='Topic', loc='upper right')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("TopicsVsDecades.png")
    plt.show()

# create graph of emotions vs amount of times

def graph_count_emotions(df):

    # Count each predicted emotion
    emotion_counts = df['predicted_emotion'].value_counts()

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(emotion_counts.index, emotion_counts.values, color='skyblue')

    plt.title('Amounts of Predicted Emotions')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("EmotionsVsCounts.png")

# create graph of emotions vs party
def graph_party_emotions(df):
    df['predicted_emotion'] = df['predicted_emotion'].str.strip().str.lower()
    df['Party'] = df['Party'].str.strip()

    # Group by party and emotion
    emotion_party_counts = df.groupby(['Party', 'predicted_emotion']).size().unstack(fill_value=0)

    # Plot
    emotion_party_counts.T.plot(kind='bar', figsize=(12, 6))  # Transpose so emotions are on x-axis

    plt.title('Emotion Frequency by Party')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Party')
    plt.tight_layout()
    plt.savefig("EmotionsVsPartys.png")

    df_filtered = df[df['Party'].isin(['Democratic', 'Republican'])]

    # Group by party and emotion
    emotion_party_counts = df_filtered.groupby(['Party', 'predicted_emotion']).size().unstack(fill_value=0)

    # Plot
    emotion_party_counts.T.plot(kind='bar', figsize=(12, 6))  # Transpose for emotions on x-axis

    plt.title('Emotion Frequency by Party (Democratic & Republican)')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Party')
    plt.tight_layout()
    plt.savefig("EmotionsVsOnlyDandR.png")

# create graph of emotions vs time
def graph_time_emotions(df):
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['predicted_emotion'] = df['predicted_emotion'].str.strip().str.lower()

    # Create a 'decade' column as the starting year of the decade
    df['decade'] = (df['year'] // 10) * 10

    # Group by decade and emotion
    emotion_decade = df.groupby(['decade', 'predicted_emotion']).size().reset_index(name='count')

    # Pivot to get decades as index and emotions as columns
    emotion_pivot = emotion_decade.pivot(index='decade', columns='predicted_emotion', values='count').fillna(0)

    # Plot
    plt.figure(figsize=(20, 8))
    emotion_pivot.plot(linewidth=2)

    plt.title('Emotion Trends by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Count')
    plt.xticks(ticks=emotion_pivot.index, labels=emotion_pivot.index, rotation=45, ha='right')
    plt.grid(True)
    plt.legend(title='Emotion')
    plt.tight_layout()
    plt.savefig("EmotionsVsTime.png")

# create graph per topic and of its emotions
def graphs_per_topic_of_emotions(df):
    df['predicted_emotion'] = df['predicted_emotion'].str.strip().str.lower()
    df['topics'] = df['topics'].astype(str)

    # Explode topics: split multi-topic rows into multiple rows
    df_expanded = df.assign(topic=df['topics'].str.split(',')).explode('topic')
    df_expanded['topic'] = df_expanded['topic'].str.strip()

    # Get unique topics
    unique_topics = df_expanded['topic'].dropna().unique()

    # For each topic, filter and count emotions
    for topic in unique_topics:
        topic_df = df_expanded[df_expanded['topic'] == topic]

        # Count the emotions
        emotion_counts = topic_df['predicted_emotion'].value_counts().sort_values(ascending=False)

        if emotion_counts.empty:
            continue  # Skip topics with no emotion data

        # Plot
        plt.figure(figsize=(8, 5))
        plt.bar(emotion_counts.index, emotion_counts.values, color='skyblue')
        plt.title(f"Emotions for Topic: {topic}")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save chart
        filename = f"{topic.replace(' ', '_')}_emotions.png"
        plt.savefig(filename)
        plt.close()


    df['predicted_emotion'] = df['predicted_emotion'].str.strip().str.lower()
    df['topics'] = df['topics'].astype(str)

    # Keep only rows with a single topic (no comma)
    df_single_topic = df[~df['topics'].str.contains(',')].copy()
    df_single_topic['topic'] = df_single_topic['topics'].str.strip()

    # Get unique single topics
    unique_topics = df_single_topic['topic'].dropna().unique()

    # Generate bar chart for each single topic
    for topic in unique_topics:
        topic_df = df_single_topic[df_single_topic['topic'] == topic]

        emotion_counts = topic_df['predicted_emotion'].value_counts().sort_values(ascending=False)

        if emotion_counts.empty:
            continue

        plt.figure(figsize=(8, 5))
        plt.bar(emotion_counts.index, emotion_counts.values, color='salmon')
        plt.title(f"Emotions for Topic: {topic} (Single-topic rows only)")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        filename = f"only_{topic.replace(' ', '_')}_emotions.png"
        plt.savefig(filename)
        plt.close()


df = pd.read_excel('speeches_with_topics_new.xlsx')
df = df[df['topics'] != 'None']
graph_count_topics(df)
graph_party_topic(df)
graph_time_topics(df)
df = pd.read_excel('speeches_with_emotions_final.xlsx')
graph_count_emotions(df)
graph_party_emotions(df)
graph_time_emotions(df)
df = pd.read_excel('combined_emotion_and_topic.xlsx')
df = df[df['topics'] != 'None']
graphs_per_topic_of_emotions(df)