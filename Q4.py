from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd


def calculate_date_difference(date_string1, date_string2):
    format_string_checkin_date = '%Y-%m-%d %H:%M:%S'
    format_string_cancellation_datetime = '%Y-%m-%d'
    try:
        date1 = datetime.strptime(date_string1, format_string_checkin_date).replace(hour=0, minute=0, second=0)
        date2 = datetime.strptime(date_string2, format_string_cancellation_datetime).replace(hour=0, minute=0, second=0)
        difference = date1 - date2

    except (ValueError, TypeError):
        difference = timedelta(days=-1)

    return difference.days  # Return the number of days as an integer


def calculate_date_difference_days_ahead(date_string1, date_string2):
    format_string = '%Y-%m-%d %H:%M:%S'
    try:
        date1 = datetime.strptime(date_string1, format_string).replace(hour=0, minute=0, second=0)
        date2 = datetime.strptime(date_string2, format_string).replace(hour=0, minute=0, second=0)
        difference = date2 - date1

    except (ValueError, TypeError):
        difference = -1

    return difference.days if difference.days >= 0 else -1


if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    df['cancel_before_checking'] = df.apply(lambda row: calculate_date_difference(row['checkin_date'],
                                                                                  row['cancellation_datetime']), axis=1)
    df['days_ahead'] = df.apply(
        lambda row: calculate_date_difference_days_ahead(row['booking_datetime'], row['checkin_date']), axis=1)

    # Specify the bin edges
    bin_edges = [0, 1, 7, 14, 30, 60, 90, 180, 365]

    # Group the data into bins based on 'cancel_before_checking' column
    cancel_bins = pd.cut(df['cancel_before_checking'], bins=bin_edges)
    all_bins = pd.cut(df['days_ahead'], bins=bin_edges)
    # Count the occurrences in each bin
    cancel_counts = cancel_bins.value_counts().sort_index()
    all_counts = all_bins.value_counts().sort_index()

    # Plot the histogram
    cancel_counts.plot(kind='pie' ,autopct='%1.1f%%')
    # Set labels and title
    # plt.xlabel('Cancel Before Checking (Days)')
    # plt.ylabel('Count')
    plt.title('Cancellation Count by Days Before Checking')
    # Display the plot
    plt.tight_layout()

    plt.show()

    all_counts.plot(kind='bar')
    plt.xlabel('All orders Before Checking (Days)')
    plt.ylabel('Count')
    plt.title('All Count by Days Before Checking')
    # Display the plot
    plt.tight_layout()

    plt.show()

    new_bins = [0, 1, 3, 7, 14, 30, 90, 180, 365]

    # Calculate the cumulative sum of the counts
    cancel_cumulative = cancel_counts.cumsum()

    # Plot the cumulative bar plot
    cancel_cumulative.plot(kind='bar')

    # Set labels and title
    plt.xlabel('Cancel Before Checking (Days)')
    plt.ylabel('Cumulative Count')
    plt.title('Accumulation Differences by Days Before Checking')
    plt.tight_layout()

    plt.show()
    cumulative_sum = all_counts[::-1].cumsum()[::-1]
    # cumulative_sum_cancels = cancel_counts[::-1].cumsum()[::-1]

    # Calculate the percentage of cancellation relative to the cumulative sum of all bins above it
    cancel_percentage = ((cancel_counts)/ cumulative_sum) * 100

    # Plot the percentage bar plot
    cancel_percentage.plot(kind='bar')

    # Set labels and title
    plt.xlabel('Cancel Before Checking (Days)')
    plt.ylabel('Cancellation Percentage')
    plt.title('Cancellation Percentage by Days Before Checking (Cumulative)')

    # Display the plot
    plt.tight_layout()

    plt.show()

    policies_counts = df['cancellation_policy_code'].value_counts().head(10)

    # Plot the histogram
    policies_counts.plot(kind='bar')

    # Set labels and title
    plt.xlabel('Cancellation Policy')
    plt.ylabel('Count')
    plt.title('Top Ten Cancellation Policies')

    # Display the plot
    plt.tight_layout()
    plt.show()