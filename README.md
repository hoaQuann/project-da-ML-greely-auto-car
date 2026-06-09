# project-da-ML-greely-auto-car
# 📊 Machine Learning - Car Price Prediction: Forecasting Greely Auto Car Prices

## I. Context
*   **Overall Objective:** Implement a complete Machine Learning pipeline in Python to forecast car prices for the Greely Auto brand.
*   **Specific Goal:** Identify the correlation between car technical specifications (features) and the final selling price (target). Develop a visualization function to compare discrepancies between AI-predicted prices and actual prices.
*   **Context / Problem Statement (5W1H):**
    *   **Why is there a problem:** Mitigate the risk of inaccurate pricing (overpricing leading to inventory buildup, underpricing reducing profit margins) through a data-driven predictive model. Traditional pricing processes are often subjective and lack quantitative insights into key value drivers.
    *   **What:** This project addresses a Regression problem aimed at predicting a continuous value (car prices in USD) based on a dataset of Greely car configurations and technical specifications.
    *   **Why:** There is a need for a quantitative statistical tool to evaluate which factors (e.g., Horsepower, Fuel Type, Dimensions, Engine Size, etc.) truly determine car prices, moving beyond subjective pricing methods.
    *   **How:** The solution involves a structured analytics workflow: Data Loading -> Data Cleaning (handling null values, outliers) -> Exploratory Data Analysis (identifying causal relationships) -> Model Training (predicting outcomes) -> Error Evaluation (Residuals analysis).
    *   **Where / When:** Analysis and coding will be performed in a Python environment (Google Colab/Jupyter Notebook). The model and insights will be utilized for evaluating new car model prices before market launch, or for building a dynamic pricing system for used cars.
*   **Stakeholders & Roles (Who):**
    *   **Data Analyst (You - Quân):** Responsible for code optimization, data processing (cleaning, transformation), machine learning model development, and creating error visualization charts.
    *   **Sales Team (End-users):** Will use the model's predictions and error visualizations to inform and adjust their pricing strategies for various car models.
    *   **Board of Directors (Stakeholders):** Will leverage the model's insights to make strategic decisions regarding product positioning, market entry pricing, and overall brand profitability.

## II. Domain Knowledge & Terminology
*   **Regression:** A supervised machine learning task that predicts a continuous output variable based on input features. In this project, it's used to predict car prices.
*   **Machine Learning Pipeline:** A series of interconnected steps that process data, train a model, and generate predictions, ensuring reproducibility and consistency.
*   **Features / Technical Specifications:** Independent variables or attributes of a car (e.g., horsepower, engine size, fuel type, body style, dimensions) used as inputs to predict the price.
*   **Target Variable:** The dependent variable that the model aims to predict, which in this case is the car's selling price in USD.
*   **Residuals:** The difference between the actual observed values and the values predicted by the model. Analyzing residuals helps evaluate model performance and identify areas for improvement.
*   **Overpricing/Underpricing Risk:** The business risk associated with setting car prices too high (leading to unsold inventory) or too low (resulting in lost profit margins).

## III. Main Content

### 1. Data Overview & Analytical Objectives
*   **Dataset:** This project utilizes a proprietary dataset containing Greely car configurations and technical specifications. The dataset is expected to include various attributes relevant to car pricing.
*   **Objectives:** Answer the following key questions:
    1.  What is the optimal selling price for a Greely Auto car based on its technical specifications?
    2.  Which technical specifications (e.g., horsepower, engine size, fuel economy) have the most significant impact on a car's price?
    3.  How accurately can a machine learning model predict car prices, and what are the primary sources of prediction error?
    4.  How can model predictions be used to mitigate the risks of overpricing and underpricing in Greely Auto's sales strategy?

### 2. Data Dictionary
A detailed data dictionary will be generated upon loading the specific dataset. Below is an example structure for anticipated columns:

| **Column Name** | **Description** | **Data Type** | **Encoding / Notes** |
| :---------------- | :---------------------------------------------- | :------------ | :------------------------------------------------------------------------------------------------- |
| `car_ID` | Unique identifier for each car | `Integer` | Primary Key. Should be **dropped** before Machine Learning modeling. |
| `symboling` | Insurance risk rating | `Integer` | +3 indicates high risk, -3 indicates safe. Ordinal nature. |
| `CarName` | Name of the car company and model | `String` | High Cardinality. Needs Feature Engineering (e.g., split into `Brand` and `Model`) to be useful for ML. |
| `wheelbase` | Distance between front and rear axles | `Float` | Continuous numerical feature. Needs scaling (Standard/MinMax). |
| `carlength` | Overall length of the car | `Float` | Continuous numerical feature. Needs scaling. |
| `carwidth` | Overall width of the car | `Float` | Continuous numerical feature. Needs scaling. |
| `carheight` | Overall height of the car | `Float` | Continuous numerical feature. Needs scaling. |
| `curbweight` | Weight of the car without occupants or baggage | `Integer` | Heavier cars usually cost more. Needs scaling. |
| `doornumber` | Number of doors | `String (Category)` | Values: 'two', 'four'. Convert to Integer (2, 4) or use Binary Encoding. |
| `carbody` | Type of car body | `String (Category)` | Values: convertible, hatchback, sedan, wagon, hardtop. Use One-Hot Encoding. |
| `enginetype` | Type of engine | `String (Category)` | Values: OHC, DOHC, ohcv, etc. Use One-Hot Encoding. |
| `cylindernumber` | Number of cylinders | `String (Category)` | Values: four, six, eight, twelve, etc. Convert from words to Integers (4, 6, 8, 12). |
| `fuelsystem` | Fuel injection/delivery system | `String (Category)` | Values: mpfi, 2bbl, mfi, etc. Use One-Hot Encoding. |
| `fueltype` | Type of fuel used | `String (Category)` | Values: 'gas' (gasoline) or 'diesel'. Use Binary Encoding (0/1). |
| `aspiration` | Engine aspiration method | `String (Category)` | Values: 'std' (standard) or 'turbo'. Use Binary Encoding (0/1). |
| `enginelocation` | Location of the engine in the car | `String (Category)` | Values: 'front' or 'rear'. Use Binary Encoding. Note: 'rear' is rare and correlates with high price. |
| `drivewheel` | Type of drive wheel | `String (Category)` | Values: 'fwd' (front), 'rwd' (rear), '4wd' (4-wheel). Use One-Hot Encoding. |
| `enginesize` | Size/displacement of the engine | `Integer` | **Strong positive correlation** with price expected. Needs scaling. |
| `boreratio` | Stroke to bore ratio of the cylinders | `Float` | Continuous numerical feature. Needs scaling. |
| `stroke` | Volume of the engine (stroke) | `Float` | Continuous numerical feature. Needs scaling. |
| `compressionratio` | Compression ratio of the engine | `Float` | Continuous numerical feature. Needs scaling. |
| `horsepower` | Engine horsepower | `Integer` | **Strong positive correlation** with price expected. Needs scaling. |
| `peakrpm` | Peak revolutions per minute | `Integer` | Continuous numerical feature. Needs scaling. |
| `citympg` | Miles per gallon in city driving | `Integer` | Fuel efficiency. Usually has a **negative correlation** with price/weight. Needs scaling. |
| `highwaympg` | Miles per gallon on highway driving | `Integer` | Fuel efficiency. Needs scaling. |
| `price` | Price of the car | `Float` | **Target Variable (Label).** Continuous numeric value for Regression modeling. |

### 3. Feature Engineering based on Domain Knowledge
This step will involve creating new features from existing ones to potentially improve model performance and capture more complex relationships within the data.
*   **`power_to_weight_ratio`:** Calculated as `horsepower` / `curb_weight`. *Reason:* This ratio is a strong indicator of a vehicle's performance and is often correlated with higher pricing, especially in performance-oriented models.
*   **`fuel_efficiency_index`:** A combined metric from `city_mpg` and `highway_mpg`. *Reason:* Represents the overall fuel economy, which is a significant factor for consumers and thus influences price.
*   **`luxury_index`:** Could be a composite score derived from features like `engine_size`, `horsepower`, and `body_style` (e.g., 'sedan' vs. 'SUV'). *Reason:* Captures the perceived luxury or premium segment of the vehicle, which directly impacts its market value.

### 4. Analytics Workflow

#### Step 1: Data Loading & Pre-processing
*   **Tools:** Python (`pandas`, `numpy`)
*   **Details:**
    *   Load the raw car configuration dataset into a pandas DataFrame.
    *   Initial inspection of data types, missing values, and summary statistics.
    *   Handle missing values: Depending on the column and extent of missingness, strategies may include imputation (mean, median, mode), or removal of rows/columns.
    *   Address duplicate entries to ensure data integrity.
    *   Convert categorical features (e.g., `fuel_type`, `body_style`, `num_doors`) into numerical representations using techniques like One-Hot Encoding or Label Encoding.

#### Step 2: Exploratory Data Analysis (EDA)
*   **Tools:** Python (`matplotlib`, `seaborn`, `pandas_profiling` or `sweetviz` for automated reports)
*   **Details:**
    *   **Univariate Analysis:** Visualize distributions of individual features (histograms, box plots) to understand their spread, central tendency, and identify outliers.
    *   **Bivariate Analysis:** Explore relationships between independent variables and the target variable (`price`) using scatter plots and correlation matrices. This will help identify key drivers of car prices.
    *   **Multivariate Analysis:** Investigate relationships among multiple variables to uncover more complex patterns.
    *   **Outlier Handling:** Identify and address outliers using statistical methods (e.g., IQR rule, Z-score) or domain-specific thresholds. Outliers may be capped, transformed, or removed, depending on their nature and impact.
    *   **Initial Insights:** Summarize preliminary findings on the most influential features and potential challenges for modeling.

#### Step 3: Data Modeling & Evaluation
*   **Tools:** Python (`scikit-learn`, `numpy`)
*   **Details & Techniques:**
    *   **Feature Scaling:** Apply techniques like StandardScaler or MinMaxScaler to normalize numerical features, which is crucial for many machine learning algorithms.
    *   **Model Selection:** Experiment with various regression algorithms, such as Linear Regression, Ridge/Lasso Regression, Decision Trees, Random Forest, Gradient Boosting (e.g., XGBoost, LightGBM), and Support Vector Machines.
    *   **Model Training:** Split the dataset into training and testing sets to evaluate model generalization. Train selected models on the training data.
    *   **Hyperparameter Tuning:** Use techniques like GridSearchCV or RandomizedSearchCV to optimize model hyperparameters for improved performance.
    *   **Model Evaluation:** Assess model performance using metrics relevant to regression tasks, such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2 score).
    *   **Residual Analysis:** Analyze the residuals (differences between predicted and actual prices) using scatter plots and histograms to check for bias, heteroscedasticity, or other patterns that indicate model shortcomings.

#### Step 4: Data Visualization
*   **Tools:** Python (`matplotlib`, `seaborn`), potentially interactive libraries like `Plotly` for enhanced exploration.
*   **Details:**
    *   **Layout Strategy:** Visualize information following a General -> Specific approach, moving from overall trends to detailed breakdowns. Use Left -> Right and Cause -> Effect arrangements where appropriate for logical flow.
    *   **Chart Selection:**
        *   **Scatter Plot of Predicted vs. Actual Prices:** To visually assess the model's accuracy and identify where predictions deviate significantly from actual values.
        *   **Histogram of Residuals:** To check if the model's errors are normally distributed around zero, indicating a well-performing model with unbiased predictions.
        *   **Feature Importance Bar Chart:** To highlight which technical specifications are most influential in predicting car prices, providing actionable insights for product development and marketing.
        *   **Correlation Heatmap:** To visualize relationships between all features and the target variable, aiding in understanding data structure.
        *   **Custom Visualization Function:** A dedicated function will be created to allow end-users to easily compare AI-predicted prices against actual prices for individual car models or segments, potentially displaying error margins.

### 5. Conclusions & Actionable Recommendations
*   **Key Findings:**
    *   The machine learning model successfully identifies and quantifies the impact of various technical specifications (e.g., `horsepower`, `engine_size`, `body_style`, `curb_weight`) on Greely Auto car prices.
    *   The model achieves a [specific R2 score/RMSE value] indicating a [strong/moderate] predictive capability for car prices.
    *   Analysis of feature importance reveals that [Top 2-3 significant features, e.g., 'horsepower' and 'luxury_index'] are the strongest determinants of car value, suggesting these areas warrant particular attention in product design and marketing.
    *   The residual analysis indicates [e.g., minimal bias / slight underprediction in high-end cars], providing specific areas for model refinement.
*   **Actionable Recommendations:**
    *   **For the Sales Team:** Utilize the predictive model and its visualizations to generate data-driven price recommendations for new car models and to quickly adjust pricing for used cars. This will help optimize inventory turnover and maximize profit margins by avoiding overpricing and underpricing.
    *   **For the Product Development Team:** Focus R&D efforts on enhancing features identified as highly influential in driving car prices (e.g., optimizing engine performance, introducing new luxury amenities) to align with market demand and pricing potential.
    *   **For the Marketing Team:** Emphasize the key technical specifications that significantly impact car value in marketing campaigns, effectively communicating the value proposition to potential buyers.
    *   **For the Board of Directors:** Integrate this data-driven pricing approach into the overall business strategy to support market competitiveness, improve profitability, and make informed decisions on new model launches and market positioning. Regular model retraining and performance monitoring are recommended to adapt to evolving market conditions.
