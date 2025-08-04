import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def grpah_time_topics(df):
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


df = pd.read_excel('speeches_with_topics.xlsx')
df = df[df['topics'] != 'None']
graph_count_topics(df)
graph_party_topic(df)
grpah_time_topics(df)