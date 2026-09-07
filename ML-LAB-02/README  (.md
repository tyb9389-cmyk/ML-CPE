# LAB02 Data Preprocessing 
LAB02\ML-LAB-02.docx
Data preprocessing is a crucial step that prepares data for practical use, enabling AI/ML models to perform efficiently and deliver the best possible results.

## Dataset
[Automobile Market Analytics Dataset](https://www.kaggle.com/datasets/deeplumiere/automobile-market-analytics-dataset)
automobile_dataset.csv (5500 rows, 18 columns)
A synthetic automobile dataset containing vehicle specifications, ownership hist

## Topic
- Part 1: Data Exploration
    - Load Dataset 
        - Load the file with pd.read_csv() and view the .head()
    - Display Shape 
        - 5500 row, 18 columns
    - Display Data Types 
        - Check the data type using df.info() (9-column number, 9-column text)
    - Display Summary Statistics 
        - Check summary using df.describe()
    - Display Missing Values 
        - Check missing values using df.isnull().sum()
        - find 9 columns missing values
    - Display Duplicate Records 
        - Check duplicate using df.duplicated().sum()
        - No duplicate entries found
    - Display Class Distribution 
        - Check class distributions of Fuel_Type, Make, Transmission, and Body_Type using value_counts().
- Part 2: Data Visualization
    - Histogram
    - Correlation Heatmap 
- Part 3: Data Cleaning
    - Missing Value Handling 
        - Fill numeric missing with median
        - Fill categorical missing with mode
    - Duplicate Removal  
        - Dont have any duplicates
    - Incorrect Data Correction
        - Checking for negative values ​
        - Checking for incorrect Year ranges (<1980 or >2026)
        - Checking Engine_Size=0 for non-electric vehicles
    - Data Type Conversion
        - Convert Accident_History from float to int
        - Convert text columns to category
- Part 4: Feature Engineering
    - Label Encoding 
        - Model (40 values), Transmission (2 values)
    - One-Hot Encoding
        - Make, Fuel_Type, Color, Body_Type, Drivetrain, Location (total 41 new columns)
    - Ordinal Encoding  
        - Service_History by order No Service(0) < Partial Service(1) < Full Service(2)

Final column count: 18 (original) + 2 (label) + 41 (one-hot) + 1 (ordinal) = 62 columns
Output : 'automobile_dataset_cleaned.csv'(5500,62)
