from war_analysis_helper import *
import re
from typing import List, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_use_of_keyword_over_periods(df: pd.DataFrame,
                                     keywords: List[str],
                                     periods: List[Dict],
                                     normalize_value,
                                     y_axis_limit,
                                     plot_title,
                                     date_column: str = 'date',
                                     text_column: str = 'speech',
                                     output_path: str = None):
    """
    Create a bar plot - each bar represents a time period and measures the number of keywords appear in the speeches
    of that time.
    :param df: A dataframe with a date column and a speech column.
    :param keywords: A set of keywords to count their appearances.
    :param periods: A dictionary of the name of the period and a range of year, e.g. {"ww2": (1939, 1945)}.
    :param normalize_value: A parameter to normalize the color of the bar
    (best set as the highest value of keywords/speech).
    :param y_axis_limit: A limit to adjust the bars heights.
    :param plot_title: Title of the plot.
    :param date_column: Name of date column.
    :param text_column: Name of text column.
    :param output_path: To save the fig.
    :return:
    """
    # Converting date columns to date format and dropping columns
    df = df.dropna(subset=[date_column, text_column]).copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    keywords_set = set(word.lower() for word in keywords)

    def count_keywords_in_text(text):
        """
        Counting keywords in speech.
        :param text: Speech.
        :return: Keyword count.
        """
        words = re.findall(r'\b\w+\b', str(text).lower())
        return sum(1 for word in words if word in keywords_set)

    # Counting keywords per speech
    df['keywords_count'] = df[text_column].apply(count_keywords_in_text)

    period_labels = []
    keywords_counts = []
    speech_counts = []
    normalized_counts = []

    for period in periods:
        start, end = list(period.values())[0]
        # Filtering speeches based on years of period and summing counts
        period_df = df[(df[date_column].dt.year >= start) & (df[date_column].dt.year <= end)]
        total_keywords_in_period = period_df['keywords_count'].sum()
        total_speeches = len(period_df)
        # Normalizing number of keywords counts per number of speeches
        norm = total_keywords_in_period / total_speeches if total_speeches > 0 else 0

        period_labels.append(f"{list(period.keys())[0]}\n({start}-{end})")
        keywords_counts.append(total_keywords_in_period)
        speech_counts.append(total_speeches)
        normalized_counts.append(norm)

    x = np.arange(len(periods))
    width = 0.6

    # Absolute scaling: clip at 10 for color intensity, scale to 0-1 for colormap
    norm_vals = np.clip(normalized_counts, 0, normalize_value) / normalize_value

    colors = plt.cm.Greens(norm_vals)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, keywords_counts, width, color=colors)

    ax.set_ylim(0, y_axis_limit)

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        label = (
            f'{height}\n'
            f'({speech_counts[idx]} speeches)\n'
            f'{normalized_counts[idx]:.2f} per speech'
        )
        ax.text(bar.get_x() + bar.get_width() / 2, height + 5,
                label,
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(period_labels)
    ax.set_ylabel(Y_LABEL)
    ax.set_xlabel(X_LABEL)
    plt.suptitle(plot_title, fontsize=15)
    ax.text(
        0.01, 1.02,
        WARS_COMPARISON_CAPTION,
        transform=ax.transAxes,
        fontsize=12, color="gray", ha="left", va="bottom"
    )
    plt.tight_layout()

    # Save and show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()


def add_war_labels(ax, df, title, start_year, end_year):
    """
    Add a label for a war period in the middle of the time range.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to annotate.
        df (DataFrame): Grouped dataframe with 'year', 'war_avg', 'peace_avg'.
        title (str): Name of the war.
        start_year (int): Start year of the war.
        end_year (int): End year of the war.
    """
    mid_year = (start_year + end_year) / 2

    # Compute the peak value in that period to place the text
    period_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    if not period_df.empty:
        peak_val = max(period_df['war_avg'].max(), period_df['peace_avg'].max())
        ax.text(
            mid_year,
            peak_val + 0.7,  # slightly above the peak
            title,
            fontsize=9,
            ha='center',
            va='bottom',
            fontweight='bold'
        )


def plot_keyword_trends_per_year(df: pd.DataFrame,
                                 start_year: int,
                                 end_year: int,
                                 war_keywords: List[str],
                                 peace_keywords: List[str],
                                 war_periods: List[Tuple[str, int, int]],
                                 date_column: str = 'date',
                                 text_column: str = 'speech',
                                 normalize_per_speech: bool = True,
                                 rolling_window: int = 3,
                                 output_path: str = None):
    """
    Plot per-year usage of two keyword sets in speeches, with markers on war start/end years.

    :param df: DataFrame containing speeches and dates.
    :param start_year: First year to include.
    :param end_year: Last year to include.
    :param war_keywords: List of keywords for 'fighting moral'.
    :param peace_keywords: List of keywords for 'peace/end war'.
    :param war_periods: List of tuples (war_name, start_year, end_year).
    :param date_column: Name of date column.
    :param text_column: Name of speech text column.
    :param normalize_per_speech: If True, average keyword count per speech; if False, total count.
    :param rolling_window: Number of years for rolling average smoothing (set to 1 or 0 for no smoothing).
    :param output_path: Path to save the figure (optional).
    """
    # Ensure date is datetime
    df = df.dropna(subset=[date_column, text_column]).copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Lowercase sets for matching
    war_set = set(w.lower() for w in war_keywords)
    peace_set = set(w.lower() for w in peace_keywords)

    def count_keywords_in_text(text, keywords_set):
        words = re.findall(r'\b\w+\b', str(text).lower())
        return sum(1 for word in words if word in keywords_set)

    # Filter by year range
    df = df[(df[date_column].dt.year >= start_year) & (df[date_column].dt.year <= end_year)]

    # Count per speech
    df['war_count'] = df[text_column].apply(lambda x: count_keywords_in_text(x, war_set))
    df['peace_count'] = df[text_column].apply(lambda x: count_keywords_in_text(x, peace_set))

    # Group by year
    grouped = df.groupby(df[date_column].dt.year).agg(
        war_total=('war_count', 'sum'),
        peace_total=('peace_count', 'sum'),
        speeches=('speech', 'count')
    ).reset_index().rename(columns={date_column: 'year'})

    if normalize_per_speech:
        grouped['war_avg'] = grouped['war_total'] / grouped['speeches']
        grouped['peace_avg'] = grouped['peace_total'] / grouped['speeches']
    else:
        grouped['war_avg'] = grouped['war_total']
        grouped['peace_avg'] = grouped['peace_total']

    # Apply rolling average if requested
    if rolling_window and rolling_window > 1:
        grouped['war_avg'] = grouped['war_avg'].rolling(window=rolling_window, center=True, min_periods=1).mean()
        grouped['peace_avg'] = grouped['peace_avg'].rolling(window=rolling_window, center=True, min_periods=1).mean()

    # Plot lines
    fig, ax = plt.subplots(figsize=(12, 6))
    line_war, = plt.plot(grouped['year'], grouped['war_avg'], label='Fighting Moral', color='orange')
    line_peace, = plt.plot(grouped['year'], grouped['peace_avg'], label='Peace Terminology', color='blue')

    # Add vertical lines for war periods
    for war_name, start, end in war_periods:
        if start_year <= start <= end_year:
            plt.axvline(x=start, color='black', linestyle='--', linewidth=1)
        if start_year <= end <= end_year:
            plt.axvline(x=end, color='black', linestyle='--', linewidth=1)
        add_war_labels(ax, grouped, war_name, start, end)

    plt.legend(handles=[line_war, line_peace])

    fig.suptitle('Comparison of Fighting Moral and Peace Terminology in War-Related Presidential Speeches')

    ax.text(
        0.01, 1.02,
        WAR_AND_PEACE_CAPTION,
        transform=ax.transAxes,
        fontsize=10, color='gray', ha='left', va='bottom'
    )

    ax.set_xlabel('Year')
    ax.set_ylabel('Average Keywords per Speech' if normalize_per_speech else 'Total Keywords')
    ax.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
