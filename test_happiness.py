"""
This is a simple test file that tests the happiness analysis script.
This test file tests data loading, filtering, grouping, and ML functionality
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# Test I: Data Loading
def test_data_loading():
    """
    Test that the 2015 CSV file loads correctly and has expected structure
    """
    # Check file
    assert os.path.exists("Data/2015.csv"), "2015.csv file not found"

    # Load
    df = pd.read_csv("Data/2015.csv")

    # Verify loading
    assert not df.empty, "Dataset is empty"
    assert len(df) > 100, "Dataset has too few countries"

    print(f"Test 1 Passed: {len(df)} countries loaded")


def test_required_columns():
    """
    Test that dataset has all required columns
    """
    df = pd.read_csv("Data/2015.csv")

    required_columns = [
        "Country",
        "Region",
        "Happiness Score",
        "Economy (GDP per Capita)",
        "Family",
        "Health (Life Expectancy)",
        "Freedom",
    ]

    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"

    print("Test 2 Passed: All required columns are present")


def test_missing_values():
    """
    Test that there are no critical missing values
    """
    df = pd.read_csv("Data/2015.csv")

    # Check happiness score has no nulls
    assert df["Happiness Score"].notna().all(), "Missing happiness scores found"
    assert df["Country"].notna().all(), "Missing country names found"

    print("Test 3 Passed: No critical missing values")


# Edge Case Tests for Data Loading
def test_data_loading_edge_cases():
    """
    Test edge cases in data loading
    """
    # Handle missing file
    try:
        _ = pd.read_csv("Data/nonexistent.csv")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    # Handle empty DataFrame
    df = pd.read_csv("Data/2015.csv")
    empty_filter = df[df["Happiness Score"] < 0]  # Should be empty
    assert len(empty_filter) == 0, "Empty filter should return 0 rows"

    # Data type
    assert pd.api.types.is_numeric_dtype(
        df["Happiness Score"]
    ), "Happiness Score should be numeric"
    assert pd.api.types.is_object_dtype(
        df["Country"]
    ), "Country should be string/object type"

    print("Test 3A Passed: Data loading edge cases handled")


# Test II: Filtering
def test_filter_top_10():
    """
    Test filtering top 10 happiest countries
    """
    df = pd.read_csv("Data/2015.csv")

    # Get top 10
    top_10 = df.nlargest(10, "Happiness Score")

    # Verify that there are exactly 10 countries
    assert len(top_10) == 10, "Should have exactly 10 countries"

    # Verify sorting order
    scores = top_10["Happiness Score"].values
    assert all(
        scores[i] >= scores[i + 1] for i in range(len(scores) - 1)
    ), "Not sorted correctly"

    # Verify Switzerland is 1 (2015 data)
    assert top_10.iloc[0]["Country"] == "Switzerland", "Wrong top country"

    print("Test 4 Passed: Top 10 countries filtered correctly")


def test_filter_region():
    """
    Test filtering countries by region
    """
    df = pd.read_csv("Data/2015.csv")

    # Filter Western Europe
    western_europe = df[df["Region"] == "Western Europe"]

    # There should be countries in this region
    assert len(western_europe) > 0, "No Western Europe countries found"
    assert len(western_europe) < len(df), "Filter didn't work"

    # All should be Western Europe
    assert all(
        western_europe["Region"] == "Western Europe"
    ), "Filter included wrong regions"

    print(f"Test 5 Passed: Filtered {len(western_europe)} Western Europe countries")


# Edge Case Tests for Filtering
def test_filter_edge_cases():
    """
    Test edge cases in filtering
    """
    df = pd.read_csv("Data/2015.csv")

    # Non-existent region
    non_existent = df[df["Region"] == "Atlantis"]
    assert len(non_existent) == 0, "Non-existent region should return empty DataFrame"

    # top N with N > dataset size
    all_countries = df.nlargest(1000, "Happiness Score")
    assert len(all_countries) == len(
        df
    ), "Should return all countries when N > dataset size"

    # NaN values
    df_with_nan = df.copy()
    df_with_nan.loc[0, "Happiness Score"] = np.nan
    top_with_nan = df_with_nan.nlargest(5, "Happiness Score")
    assert len(top_with_nan) == 5, "Should handle NaN values in nlargest"
    assert not np.isnan(
        top_with_nan.iloc[0]["Happiness Score"]
    ), "Top result should not be NaN"

    # Duplicate happiness scores
    duplicate_scores = df[df["Happiness Score"].duplicated()]
    if len(duplicate_scores) > 0:
        print(
            f"Found {len(duplicate_scores)} countries with duplicate happiness scores"
        )

    print("Test 5A Passed: Filtering edge cases handled correctly")


# Test III: Grouping Tests
def test_group_by_region():
    """
    Test grouping countries by region and calculating average happiness
    """
    df = pd.read_csv("Data/2015.csv")

    # Group by region and calculate mean
    regional_happiness = df.groupby("Region")["Happiness Score"].mean()

    # Should have multiple regions
    assert len(regional_happiness) > 5, "Too few regions"
    assert len(regional_happiness) < 50, "Too many regions"

    # All values should be between 0 and 10
    assert regional_happiness.min() > 0, "Invalid minimum happiness"
    assert regional_happiness.max() < 10, "Invalid maximum happiness"

    print(f"Test 6 Passed: Grouped into {len(regional_happiness)} regions")


# Edge Case Tests for Grouping
def test_grouping_edge_cases():
    """
    Test edge cases in grouping operations
    """
    df = pd.read_csv("Data/2015.csv")

    # Grouping with missing regions
    df_with_missing = df.copy()
    df_with_missing.loc[0, "Region"] = None
    _ = df_with_missing.groupby("Region", dropna=False)["Happiness Score"].mean()

    # Single country regions
    region_counts = df.groupby("Region").size()
    single_country_regions = region_counts[region_counts == 1]
    if len(single_country_regions) > 0:
        print(f"Found {len(single_country_regions)} regions with only one country")

    # Aggregate multiple statistics
    regional_stats = df.groupby("Region")["Happiness Score"].agg(
        ["mean", "std", "count"]
    )
    assert len(regional_stats) > 0, "Regional statistics aggregation failed"
    assert "mean" in regional_stats.columns, "Mean calculation missing"

    # No empty groups
    all_regions = df["Region"].unique()
    for region in all_regions:
        region_data = df[df["Region"] == region]
        assert len(region_data) > 0, f"Region {region} should not be empty"

    print("Test 6A Passed: Grouping edge cases handled correctly")


# Test IV: Machine Learning
def test_ml_training():
    """Test that ML model can be trained"""
    df = pd.read_csv("Data/2015.csv")

    # Prepare features and target
    features = [
        "Economy (GDP per Capita)",
        "Family",
        "Health (Life Expectancy)",
        "Freedom",
    ]
    X = df[features]
    y = df["Happiness Score"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model should have coefficients
    assert len(model.coef_) == len(features), "Wrong number of coefficients"
    assert model.intercept_ is not None, "No intercept learned"

    print("Test 7 Passed: ML model trained successfully")


def test_ml_predictions():
    """
    Test that ML model makes reasonable predictions
    """
    df = pd.read_csv("Data/2015.csv")

    features = [
        "Economy (GDP per Capita)",
        "Family",
        "Health (Life Expectancy)",
        "Freedom",
    ]
    X = df[features]
    y = df["Happiness Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Check predictions are reasonable
    assert len(predictions) == len(y_test), "Wrong number of predictions"
    assert predictions.min() > 0, "Predictions too low"
    assert predictions.max() < 10, "Predictions too high"

    # Check R² score
    score = r2_score(y_test, predictions)
    assert score > 0.5, f"Model accuracy too low: {score}"

    print(f"Test 8 Passed: Model R² score = {score:.3f}")


# Test V: System/Integration
def test_complete_analysis_pipeline():
    """
    System test: Run complete analysis pipeline
    """
    # Load data
    df = pd.read_csv("Data/2015.csv")
    assert not df.empty

    # Filter top countries
    top_10 = df.nlargest(10, "Happiness Score")
    assert len(top_10) == 10

    # Group by region
    regional = df.groupby("Region")["Happiness Score"].mean()
    assert len(regional) > 0

    # Train ML model
    features = [
        "Economy (GDP per Capita)",
        "Family",
        "Health (Life Expectancy)",
        "Freedom",
    ]
    X = df[features]
    y = df["Happiness Score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))

    assert score > 0.5

    print("Test 9 Passed: Complete pipeline works end-to-end")


def test_multi_year_comparison():
    """
    System test: Can load and compare multiple years
    """
    years_found = []

    for year in [2015, 2016, 2017, 2018, 2019]:
        filepath = f"Data/{year}.csv"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            years_found.append(year)
            assert len(df) > 100, f"Year {year} has too few countries"

    assert len(years_found) >= 1, "No data files found"
    print(f"Test 10 Passed: Found data for years: {years_found}")


def test_data_consistency():
    """
    Test data consistency across multiple years
    """
    years_data = {}

    for year in [2015, 2016, 2017, 2018, 2019]:
        filepath = f"Data/{year}.csv"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            years_data[year] = df

    if len(years_data) > 1:
        # Common countries across years
        all_countries = set()
        year_countries = {}

        for year, df in years_data.items():
            # Handle different column names
            country_col = "Country" if "Country" in df.columns else "Country or region"
            countries = set(df[country_col].values)
            year_countries[year] = countries
            all_countries.update(countries)

        # Find countries present in all years
        common_countries = all_countries.copy()
        for countries in year_countries.values():
            common_countries = common_countries.intersection(countries)

        print(f"Found {len(common_countries)} countries present in all available years")

        # Happiness score ranges consistency
        happiness_ranges = {}
        for year, df in years_data.items():
            happiness_col = (
                "Happiness Score" if "Happiness Score" in df.columns else "Score"
            )
            if happiness_col in df.columns:
                happiness_ranges[year] = (
                    df[happiness_col].min(),
                    df[happiness_col].max(),
                )

        if happiness_ranges:
            min_range = min(r[0] for r in happiness_ranges.values())
            max_range = max(r[1] for r in happiness_ranges.values())
            print(
                f"Happiness score range across all years: {min_range:.2f} - {max_range:.2f}"
            )

    print("Test 10A Passed: Multi-year data consistency check completed")


# Run all tests
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Run Happiness Analysis Tests")
    print("=" * 50 + "\n")

    test_functions = [
        test_data_loading,
        test_required_columns,
        test_missing_values,
        test_data_loading_edge_cases,
        test_filter_top_10,
        test_filter_region,
        test_filter_edge_cases,
        test_group_by_region,
        test_grouping_edge_cases,
        test_ml_training,
        test_ml_predictions,
        test_complete_analysis_pipeline,
        test_multi_year_comparison,
        test_data_consistency,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"{test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"{test_func.__name__} ERROR: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed.")
    else:
        print("Some tests failed: check above for details.")
    print("=" * 50 + "\n")

    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)
