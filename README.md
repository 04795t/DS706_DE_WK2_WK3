# DS706_DE_WK2_WK3_WK5

# World Happiness Report Analysis - 2015 and simple analysis on years 2015-2019

## Status Badge
[![World Happiness Analysis CI](https://github.com/04795t/DS706_DE_WK2_WK3/actions/workflows/main.yml/badge.svg)](https://github.com/04795t/DS706_DE_WK2_WK3/actions/workflows/main.yml)

## Project Overview

This project performs a comprehensive analysis of the World Happiness Report 2015-2019 dataset to explore global happiness patterns and the factors that contribute to happiness across 158 countries. 

I am generally interested in happiness, and wonders about the difference in happiness level of individuals and groups. As I searched for datasets, I came across this dataset, a world happiness report. I wanted to know more about happiness globally, and how these may be informational towards possible social welfare decisions. There are many factors included, and I was only able to perform a simple analysis, but it was still interesting to see the rankings and the factors that are impactful towards national happiness. 

The analysis includes:

* Data inspection and cleaning
* Regional and country-level filtering and grouping
* Machine learning modeling to predict happiness scores
* Visualizations summarizing global happiness trends

---

## Project Goals

1. To understand global happiness trends across regions and countries.
2. To identify key happiness factors such as GDP, family/social support, health, and freedom.
3. To build predictive models for happiness scores using linear regression.
4. To visualize insights to communicate patterns effectively.

---

## Dataset Information

* **Source**: [Kaggle - World Happiness Report](https://www.kaggle.com/datasets/unsdsn/world-happiness)
* **Files**: `2015.csv`,`2016.csv`,`2017.csv`,`2018.csv`,`2019.csv`
* **2015 is the main focus of this analysis, but it can be easily changed to analyze any year**
* **Countries included in 2015**: 158
* **Features**: 12

### Key Variables in the Dataset

| Variable                      | Description                               |
| ----------------------------- | ----------------------------------------- |
| Country                       | Name of the country                       |
| Region                        | Geographic region                         |
| Happiness Rank                | Rank of country based on happiness score  |
| Happiness Score               | Overall happiness measure (0–10)          |
| Economy (GDP per Capita)      | Economic production contribution          |
| Family                        | Social support contribution               |
| Health (Life Expectancy)      | Life expectancy contribution              |
| Freedom                       | Freedom to make life choices              |
| Trust (Government Corruption) | Perception of corruption                  |
| Generosity                    | Generosity contribution                   |
| Dystopia Residual             | Comparison to worst-case scenario country |

---

## Setup Instructions

### Prerequisites

* Python 3.8+
* pip

### Installation

1. Clone the repository:

```bash
git clone https://github.com/04795t/DS706_DE_WK2_WK3.git
```

2. Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

3. Place the `2015.csv`, `2016.csv`, `2017.csv`, `2018.csv`, `2019.csv` dataset in the project folder.

### Running the Analysis

```bash
python happiness_analysis.py
python comparison_analysis.py
```
**Important Note: When running the scripts, visualization windows will pop up displaying the charts. You must close these image windows for the script to complete execution and proceed to the next steps.**

---

## Containerization

### Docker

#### Prerequisites
* Docker Desktop installed and running

#### Building and Running
1. Build the Docker image:
```bash
docker build -t happiness-analysis .
```

2. Run tests in Docker:
```bash
docker run happiness-analysis
```

3. Run analysis scripts in Docker:
```bash
docker run happiness-analysis python happiness_analysis.py
docker run happiness-analysis python comparison_analysis.py
```

### VS Code Dev Container

#### Prerequisites
* Visual Studio Code
* Docker Desktop installed
* "Dev Containers" extension in VS Code

#### Setup and Running
1. Open project in VS Code:
```bash
code DS706_DE_WK2_WK3
```

2. When prompted, click "Reopen in Container" (or press command+shift+p and select "Dev Containers: Reopen in Container")
* The container will automatically install Python dependencies from `requirements.txt`

3. Run tests or analysis in the container terminal:
```bash
python test_happiness.py
python happiness_analysis.py
python comparison_analysis.py
```

---

## Analysis Results

### 1. Data Import & Inspection

* Loaded the CSV dataset and verified the size, there are 158 countries and 12 features
* Inspected first rows (.head()), data types (.info()), and descriptive statistics (.describe())
* Checked for missing values and duplicates: none were found
* Happiness score range from 2.839 to 7.587

### 2. Filtering & Grouping

* **Regional Analysis**: Group countries by `Region` to calculate average happiness scores.

 * **Top Regions**:
    1. Australia and New Zealand: 7.285
    2. North America: 7.273
    3. Western Europe: 6.690
    4. Latin America and Caribbean: 6.145
    5. Eastern Asia: 5.626

* **Top 10 Countries**:
  1. Switzerland: 7.587
  2. Iceland: 7.561
  3. Denmark: 7.527
  4. Norway: 7.522
  5. Canada: 7.427
  6. Finland: 7.406
  7. Netherlands: 7.378
  8. Sweden: 7.364
  9. New Zealand: 7.286
  10. Australia: 7.284

### 3. Machine Learning
* **Model**: Linear Regression
* **Features**: `Economy (GDP per Capita)`, `Family`, `Health (Life Expectancy)`, `Freedom`
* **Target**: `Happiness Score`
* Split dataset: 80% training, 20% testing
* **R² Score**: 0.836 (83.6% accuracy)
* Successfully explains 83.6% of happiness variation across countries

### 4. Visualizations
* `happiness_analysis.png`: 2015 detailed analysis including:
    * Distribution of happiness scores
    * Top 10 happiest countries bar chart
    * GDP vs Happiness scatter plot
    * Actual vs Predicted Happiness scatter plot with trend line
<img width="600" height="500" alt="happiness_analysis" src="https://github.com/user-attachments/assets/e30d732d-2571-46c0-9933-f117bb8b2f4c" />

* `happiness_trend_simple.png`: Multi-year trend analysis (2015-2019)
<img width="500" height="300" alt="happiness_trend_simple" src="https://github.com/user-attachments/assets/5eef0aa9-9cd2-4612-b00e-835caed9bd0f" />

### Multi-Year Comparison Analysis (2015-2019)
* **Year-over-Year Trends**:
  * 2015: Average happiness 5.376, Happiest: Switzerland (158 countries)
  * 2016: Average happiness 5.382, Happiest: Denmark (157 countries)
  * 2017: Average happiness 5.354, Happiest: Norway (155 countries)
  * 2018: Average happiness 5.376, Happiest: Finland (156 countries)
  * 2019: Average happiness 5.407, Happiest: Finland (156 countries)

**Global Summary Results**:
* **Overall Trend**: Global happiness increased from 5.376 to 5.407 over the 5-year period
* **Most Consistent Top Performer**: Finland was the happiest country 2 times (2018, 2019)
* **Dataset Coverage**: Country participation varied from 155-158 nations across years

---

## Key Findings

* **Average Global Happiness (2015)**: \~5.376
* **Happiest Country**: Switzerland (7.587)
* **Top 10 Countries**:
    1. Switzerland: 7.587
    2. Iceland: 7.561
    3. Denmark: 7.527
    4. Norway: 7.522
    5. Canada: 7.427
    6. Finland: 7.406
    7. Netherlands: 7.378
    8. Sweden: 7.364
    9. New Zealand: 7.286
    10. Australia: 7.284
* **2015-2019 Global Happiness**: Global happiness shows a slight upward trend over the 2015-2019 period
* **Strongest Predictors of Happiness**: GDP per capita, Family, Health
* **Model Performance**: Linear Regression R² ≈ 0.836

---

## Personal Reflection

These results challenged my initial assumptions about what would drive national happiness. I did expect economic factors to dominate. with R² ≈ 0.836, it was surprising how happiness appears to be predictable from simply four major factors, which was GDP, social support, health and freedom. Nordic countries are constanly at the top when ranking, and it was obvious that happiness was more than being economically wealthy. Strong social safety nets and personal autonomy is also important, which can be considered as nations make further decisions on what to develop. National policy decisions do matter, and we can see that through the fluctuation in happiness level rankings, with Finland becoming the happiest in 2018 and 2019. 

There still needs to be more analysis on how and why these factors impact national happiness, but this is a starting point to uncover happiness, and to possibly improve. 

---
## Testing

### Prerequisites
* Python 3.8+
* pandas, numpy, matplotlib, scikit-learn installed

### Running Tests
```bash
python test_happiness.py
```

### Tests cover:
1. Data Loading (4 tests)
  - Validates file existence, dataset size, required columns, and missing values  
  - Covers edge cases such as missing files, invalid filters, and data type checks  

2. Filtering (3 tests)
  - Validates top-N filtering, region-based filtering, and edge cases (non-existent regions, NaN handling, duplicate scores)  

3. Grouping (2 tests)
  - Region-based grouping and average scores  
  - Edge cases like missing regions, single-country regions, and multiple statistics aggregation  

4. Machine Learning (2 tests) 
  - Linear regression training and coefficient validation  
  - Prediction accuracy and R² score check (ensures >0.5, current ~0.83)  

5. System / Integration (3 tests)  
  - Full pipeline: load -> filter -> group -> ML model  
  - Multi-year comparison across datasets  
  - Consistency of happiness scores and countries across years  

### Passed Test:
<img width="521" height="379" alt="Screenshot 2025-09-16 at 21 28 03" src="https://github.com/user-attachments/assets/e032bda7-e726-4332-a935-6ed68366c618" />

---
## CI Workflow Success
<img width="1446" height="594" alt="image" src="https://github.com/user-attachments/assets/d777346c-a62c-4fb0-9649-1d9b8969e1a9" />

---
## Refactoring Commits
### Happiness Analysis
<img width="1515" height="807" alt="image" src="https://github.com/user-attachments/assets/236ae099-b564-4f08-90e7-ad8a0dceb10a" />

### Comparison Analysis
<img width="1519" height="935" alt="image" src="https://github.com/user-attachments/assets/eff470f4-b932-48fb-b6b8-70d1ffcbe6b4" />

---

## Possible Future Improvements

* Experiment with more advanced ML models (Random Forest, Gradient Boosting)
* Add interactive dashboards using Plotly or Dash
* Feature engineering (social support index, well-being index)

---

## References
Kaggle Dataset: [World Happiness Report](https://www.kaggle.com/datasets/unsdsn/world-happiness)
