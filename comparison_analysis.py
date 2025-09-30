def main():
    """
    Super Simple Happiness Comparison (2015-2019)
    This script analyzes global happiness trends from 2015 to 2019 and visualizes the average happiness per year.
    It also identifies the happiest country each year and determines which country was happiest most frequently.
    Note: This script assumes the data files are named '2015.csv', '2016.csv', etc., and are located in a 'Data' folder.
    """

    # Import necessary libraries
    import pandas as pd
    import matplotlib.pyplot as plt
    from collections import Counter

    print("=" * 50)
    print("Comparing happiness across all years")
    print("=" * 50)

    # Initialize lists
    years = []
    avg_scores = []
    top_countries = []

    # Loop through each year and collect key metrics
    def find_happiness_column(year, happiness_data):
        if "Happiness Score" in happiness_data.columns:
            happiness_col = "Happiness Score"
        elif "Happiness.Score" in happiness_data.columns:
            happiness_col = "Happiness.Score"
        elif "Score" in happiness_data.columns:
            happiness_col = "Score"
        else:
            raise ValueError(f"No happiness column found in {year}.csv")
        return happiness_col

    def find_country_column(year, df):
        if "Country" in df.columns:
            country_col = "Country"
        elif "Country or region" in df.columns:
            country_col = "Country or region"
        else:
            raise ValueError(f"No country column found in {year}.csv")
        return country_col

    def analyze_year(years, avg_scores, top_countries, find_country_column, year):
        try:
            # Load the CSV dataset for the current year
            df = pd.read_csv(f"Data/{year}.csv")

            # Determine the correct column name for Happiness Score (this accounts for naming differences across years)
            happiness_col = find_happiness_column(year, df)

            # Determine the correct column name for Country
            country_col = find_country_column(year, df)

            # Calculate the average global happiness for this year
            avg_score = df[happiness_col].mean()

            # Identify the country with the highest happiness score this year
            happiest_country = df.loc[df[happiness_col].idxmax(), country_col]

            # Store the results for plotting and summary
            years.append(year)
            avg_scores.append(avg_score)
            top_countries.append(happiest_country)

            # Print a summary for the current year
            print(f"\n{year}:")
            print(f"  Average happiness: {avg_score:.3f}")
            print(f"  Happiest country: {happiest_country}")
            print(f"  Number of countries: {len(df)}")

        except Exception as e:
            print(f"\n{year}: Could not load data ({e})")

    for year in range(2015, 2020):
        analyze_year(years, avg_scores, top_countries, find_country_column, year)

    # Visualization: Line chart of average happiness over the years
    def generate_visualization(years, avg_scores):
        plt.figure(figsize=(10, 6))

        # Plot the average happiness trend
        plt.plot(
            years, avg_scores, marker="o", linewidth=2, markersize=10, color="blue"
        )

        # Annotate each point with its average happiness value
        for year, score in zip(years, avg_scores):
            plt.text(year, score + 0.01, f"{score:.3f}", ha="center")

        # Add chart labels and title
        plt.xlabel("Year")
        plt.ylabel("Average Global Happiness Score")
        plt.title("World Happiness Trend (2015-2019)")

        # Add a light grid for readability
        plt.grid(True, alpha=0.3)

        # Set y-axis limits slightly beyond min and max for better visibility
        plt.ylim(min(avg_scores) - 0.05, max(avg_scores) + 0.05)

        # Ensure layout fits well
        plt.tight_layout()

        # Save the figure to a file
        plt.savefig("happiness_trend_simple.png")

        # Display the plot
        plt.show()

    generate_visualization(years, avg_scores)

    # Summary and Insights
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    # Compare the first and last year's average happiness
    if avg_scores[-1] > avg_scores[0]:
        print(
            f"Global happines increased from {avg_scores[0]:.3f} to {avg_scores[-1]:.3f}"
        )
    else:
        print(
            f"Global happiness decreased from {avg_scores[0]:.3f} to {avg_scores[-1]:.3f}"
        )

    # Determine which country was the happiest most often across all years
    most_frequent = Counter(top_countries).most_common(1)[0]
    print(f"{most_frequent[0]} was the happiest country {most_frequent[1]} times")

    print("\nChart saved as 'happiness_trend_simple.png'")
    print("Analysis complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
