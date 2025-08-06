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

    df_expanded = df_expanded[df_expanded['Party'].isin(['Republican', 'Democratic'])]

    # Group by party and topic, count occurrences
    counts = df_expanded.groupby(['Party', 'topic']).size().reset_index(name='count')

    # Pivot to have topics as index and parties as columns
    pivot_df = counts.pivot(index='topic', columns='Party', values='count').fillna(0)

    # Calculate total number of speeches per party (across all topics)
    total_speeches_per_party = pivot_df.sum(axis=0)

    # Calculate the percentage of speeches for each topic by party
    pivot_df_percentage = pivot_df.div(total_speeches_per_party, axis=1) * 100

    # Plot grouped bar chart for percentage
    pivot_df_percentage.plot(kind='bar', figsize=(16, 8))

    # Customize plot
    plt.title('Percentage of Speeches by Democrats and Republicans for Each Topic')
    plt.xlabel('Topic')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Party')
    plt.tight_layout()

    # Save the plot
    plt.savefig("TopicPercentageByParty.png")

# create graph of topics vs time
def graph_time_topics(df):
    plt.clf()
    df['year'] = pd.to_datetime(df['date']).dt.year

    # Expand the topics into separate rows
    df_expanded = df.assign(topic=df['topics'].str.split(',')).explode('topic')
    df_expanded['topic'] = df_expanded['topic'].str.strip()

    # Find the start and end decades
    min_year = df_expanded['year'].min()
    start_decade = (min_year // 10) * 10

    max_year = df_expanded['year'].max()
    end_decade = ((max_year // 10) + 1) * 10 - 1  # e.g. 2020 → 2029

    # Create decade bins edges: e.g. 1900, 1910, 1920, ..., 2030 (one edge beyond last bin)
    bins = np.arange(start_decade, end_decade + 2, 10)  # +2 to ensure last edge includes 2029
    labels = [f"{b}-{b + 9}" for b in bins[:-1]]

    # Create the 'decade_bin' column
    df_expanded['decade_bin'] = pd.cut(
        df_expanded['year'], bins=bins, labels=labels, right=True, include_lowest=True
    )

    # Group by 'decade_bin' and 'topic' and count the number of occurrences
    topic_trends = df_expanded.groupby(['decade_bin', 'topic']).size().reset_index(name='count')

    # Pivot the table to have topics as columns
    pivot_trends = topic_trends.pivot(index='decade_bin', columns='topic', values='count').fillna(0)

    # Calculate the total number of speeches per decade
    total_speeches_per_decade = pivot_trends.sum(axis=1)

    # Normalize each topic count by dividing by the total speeches in that decade
    pivot_trends_normalized = pivot_trends.div(total_speeches_per_decade, axis=0)

    # Select the top 6 topics based on total counts (before normalization)
    top_topics = pivot_trends.sum().sort_values(ascending=False).head(6).index

    # Filter the pivot table for the top 6 topics
    pivot_trends_top = pivot_trends_normalized[top_topics]

    # Plotting
    fig, ax = plt.subplots(figsize=(24, 8))

    pivot_trends_top.plot(ax=ax, linewidth=2)

    # Set x ticks to be at each decade start (numeric positions)
    decade_starts = bins[:-1]  # e.g. [1900, 1910, ..., 2020]
    xticks_pos = range(len(decade_starts))
    xticks_labels = [str(d) for d in decade_starts]

    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_labels, rotation=45, ha='right')

    ax.set_title('Topic Trends by Decade (Top 6 Topics, Normalized)')
    ax.set_xlabel('Decade Start Year')
    ax.set_ylabel('Fraction of Speeches')
    ax.legend(title='Topic', loc='upper right')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("TopicsVsDecades.png")

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

    # Precentage for each party

    emotion_counts = df_filtered.groupby(['Party', 'predicted_emotion']).size().unstack(fill_value=0)

    # Normalize each party's counts separately (row-wise)
    emotion_percentages = emotion_counts.div(emotion_counts.sum(axis=1), axis=0)

    yticks = np.arange(0, 1.1, 0.1)
    # Plot for Democratic
    plt.figure(figsize=(8, 5))
    emotion_percentages.loc['Democratic'].plot(kind='bar', color='skyblue')
    plt.title('Emotion Distribution - Democratic (Normalized)')
    plt.ylabel('Proportion (0 to 1)')
    plt.xlabel('Emotion')
    plt.yticks(yticks)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("DemocraticEmotionDistribution.png")

    # Plot for Republican
    plt.figure(figsize=(8, 5))
    emotion_percentages.loc['Republican'].plot(kind='bar', color='salmon')
    plt.title('Emotion Distribution - Republican (Normalized)')
    plt.ylabel('Proportion (0 to 1)')
    plt.xlabel('Emotion')
    plt.yticks(yticks)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("RepublicanEmotionDistribution.png")

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

    df['date'] = pd.to_datetime(df['date'])

    # Create a new column for the decade (rounding down the year to the nearest 10)
    df['decade'] = (df['date'].dt.year // 10) * 10

    # Filter data for Democratic and Republican parties
    df_democratic = df[df['Party'] == 'Democratic']
    df_republican = df[df['Party'] == 'Republican']

    # Function to plot emotion trends over decades for a given party (now with fractional values)
    def plot_emotion_trends_fraction(df_party, party_name, ax, global_max, decade_range):
        # Group by 'decade' and 'predicted_emotion', and count occurrences
        emotion_trends = df_party.groupby(['decade', 'predicted_emotion']).size().unstack(fill_value=0)

        # Calculate total speeches per decade for the party
        total_speeches_per_decade = emotion_trends.sum(axis=1)

        # Normalize: Calculate the fraction for each emotion per decade
        emotion_fraction = emotion_trends.div(total_speeches_per_decade, axis=0)

        # Plotting each emotion as a line on the graph
        for emotion in emotion_fraction.columns:
            ax.plot(emotion_fraction.index, emotion_fraction[emotion], label=emotion, marker='o')

        # Title and labels
        ax.set_title(f'Emotion Trends for {party_name} Party by Decade (Fraction)')
        ax.set_xlabel('Decade')
        ax.set_ylabel('Fraction of Speeches')
        ax.set_ylim(0, 1)  # Y-axis should always range from 0 to 1
        ax.set_xlim(decade_range)  # Set the same x-axis limit across all plots

        # Calculate equal x-ticks every 10 years (decades)
        tick_positions = range(decade_range[0], decade_range[1] + 1, 10)
        tick_labels = [f'{x}-{x + 9}' for x in tick_positions]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        ax.grid(True)
        ax.legend(title='Emotions', loc='upper left', bbox_to_anchor=(1, 1))

    # Combine data for both parties to get the global max value for y-axis
    emotion_trends_democratic = df_democratic.groupby(['decade', 'predicted_emotion']).size().unstack(fill_value=0)
    emotion_trends_republican = df_republican.groupby(['decade', 'predicted_emotion']).size().unstack(fill_value=0)

    # Calculate the global max value across both parties (for y-axis)
    # Since we're normalizing, global_max is always 1
    global_max = 1

    # Get the min and max decades across both parties for the x-axis
    min_decade = min(emotion_trends_democratic.index.min(), emotion_trends_republican.index.min())
    max_decade = max(emotion_trends_democratic.index.max(), emotion_trends_republican.index.max())
    decade_range = (min_decade, max_decade)

    # Combine both party data for the total (combined) plot
    df_combined = pd.concat([df_democratic, df_republican])

    # Group by decade and emotion for the combined data
    emotion_trends_combined = df_combined.groupby(['decade', 'predicted_emotion']).size().unstack(fill_value=0)

    # Calculate total speeches per decade for the combined data
    total_speeches_per_decade_combined = emotion_trends_combined.sum(axis=1)

    # Normalize: Calculate the fraction for each emotion per decade for combined data
    emotion_fraction_combined = emotion_trends_combined.div(total_speeches_per_decade_combined, axis=0)

    # Create subplots (3 rows, 1 column) for the three graphs: Democratic, Republican, and Combined
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))

    # Plot for Democratic Party
    plot_emotion_trends_fraction(df_democratic, 'Democratic', axes[0], global_max, decade_range)

    # Plot for Republican Party (with fraction)
    plot_emotion_trends_fraction(df_republican, 'Republican', axes[1], global_max, decade_range)

    # Plot for Combined (Both Parties) with fraction (now normalized)
    for emotion in emotion_fraction_combined.columns:
        axes[2].plot(emotion_fraction_combined.index, emotion_fraction_combined[emotion], label=emotion, marker='o')

    # Title and labels for combined plot
    axes[2].set_title('Combined Emotion Trends for Both Parties by Decade (Fraction)')
    axes[2].set_xlabel('Decade')
    axes[2].set_ylabel('Fraction of Speeches')
    axes[2].set_ylim(0, 1)  # Set y-limits from 0 to 1
    axes[2].set_xlim(decade_range)  # Set the same x-axis limit across all plots

    # Calculate equal x-ticks every 10 years (decades)
    tick_positions = range(decade_range[0], decade_range[1] + 1, 10)
    tick_labels = [f'{x}-{x + 9}' for x in tick_positions]

    axes[2].set_xticks(tick_positions)
    axes[2].set_xticklabels(tick_labels)

    axes[2].grid(True)
    axes[2].legend(title='Emotions', loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.savefig('BothPartiesEmotionsOverTime.png')

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

def graphs_per_president(df):
    # Create graph of topic and emotions distribution
    pd.set_option('display.max_columns', None)
    print(df.head())
    df_expanded = df.assign(topic=df['topics'].str.split(',')).explode('topic')
    df_expanded['topic'] = df_expanded['topic'].str.strip()
    df_expanded['date'] = pd.to_datetime(df_expanded['date'])
    topic_counts = df_expanded.groupby(['President', 'topic']).size().reset_index(name='count')

    # Group emotion counts per president
    emotion_counts = df.groupby(['President', 'predicted_emotion']).size().reset_index(name='count')

    # Unique presidents
    presidents = df['President'].unique()

    for president in presidents:
        # Filter topic data for president
        df_pres_topics = topic_counts[topic_counts['President'] == president].sort_values(by='count', ascending=False)

        # Filter emotion data for president
        df_pres_emotions = emotion_counts[emotion_counts['President'] == president].sort_values(by='count',
                                                                                                ascending=False)

        # Get year range for this president
        df_pres_dates = df[df['President'] == president]
        min_year = df_pres_dates['date'].dt.year.min()
        max_year = df_pres_dates['date'].dt.year.max()

        # --- Plot Topics ---
        plt.figure(figsize=(16, 8))
        plt.bar(df_pres_topics['topic'], df_pres_topics['count'], color='skyblue')
        plt.title(f"Topics Discussed by {president} ({min_year} - {max_year})")
        plt.xlabel("Topic")
        plt.ylabel("Number of Speeches")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{president} topics.png")

        # --- Plot Emotions ---
        plt.figure(figsize=(16, 8))
        plt.bar(df_pres_emotions['predicted_emotion'], df_pres_emotions['count'], color='coral')
        plt.title(f"Predicted Emotions of Speeches by {president} ({min_year} - {max_year})")
        plt.xlabel("Predicted Emotion")
        plt.ylabel("Number of Speeches")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{president} emotions.png")

# df = pd.read_excel('speeches_with_topics_different_threshold.xlsx')
# df = df[df['topics'] != 'None']
# graph_count_topics(df)
# graph_party_topic(df)
# graph_time_topics(df)
# df = pd.read_excel('speeches_with_emotions_final.xlsx')
# df = pd.read_excel('emotions_filtered_by_positivity_label.xlsx')
# df = df[df['predicted_emotion'] != 'neutral']
# graph_count_emotions(df)
# graph_party_emotions(df)
# graph_time_emotions(df)
# df = pd.read_excel('combined_emotion_and_topic_2.xlsx')
# df = df[df['topics'] != 'None']
# graphs_per_topic_of_emotions(df)

df = pd.read_excel('combined_predictions.xlsx')
df = df[df['topics'] != 'None']
df = df[df['predicted_emotion'] != 'neutral']
counts = df['President'].value_counts()

# Step 2: Get the top 10 presidents with the most rows
top_10_presidents = counts.head(10).index

# Step 3: Filter the DataFrame to only include those top 10 presidents
df_top_10 = df[df['President'].isin(top_10_presidents)]
graphs_per_president(df_top_10)