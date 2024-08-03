# Session 3 - Machine Learning and Case Studies

### Data Science Case Study: Airline Customer Churn

#### **Pre-Execution Phase**

**Objective Understanding / Product Understanding / Selecting Success Criteria**

1. **Objective Understanding:**
   - Identify the factors leading to customer churn.
   - Develop a predictive model to forecast potential churn customers.
   - Provide actionable insights to retain customers.

2. **Product Understanding:**
   - Understanding the airline services and customer touchpoints.
   - Analyze customer journey from booking to flight completion.
   - Identify high-value customers and their service usage patterns.

3. **Success Criteria:**
   - Reduction in customer churn rate.
   - Increase in customer retention rate.
   - Improvement in overall customer satisfaction and revenue.

**Response Definition:**
   - Response variable: Churn (1 if the customer has churned, 0 otherwise).

**Data Collection & Data Check:**
   - Collect historical data on customer transactions, demographics, and flight details.
   - Ensure data quality, completeness, and consistency.

#### **Approach**

**Response Variable:**
   - Churn

**Customer Selection Criteria:**
   - Customers who have flown at least 2 years in the last 4 years.
   - Customers who haven’t taken a flight in the past year.
   - Average Flight Value (AFV) > $100.
   - Minimum number of flights: 5.

**Data Variables:**

1. **Demographic Features (Customer Oriented):**
   - Age
   - Gender
   - Destination
   - Departure
   - Class
   - Average amount spent
   - Meal preference (opts for meal or not)
   - Rating
   - Average time between flights
   - Average journey time

2. **Transaction Features:**
   - Mode of payment
   - Reward points usage
   - Transaction time

3. **Flight Features:**
   - Timing of flight (morning/afternoon/evening/night)
   - Weekend or weekday flight
   - Average delay
   - Flight duration

### **Data Science Execution**

1. **Data Cleaning / Standardization:**
   - Handle missing values.
   - Standardize formats (e.g., dates, categorical variables).
   - Remove duplicates and irrelevant data.

2. **Outlier Treatment / Missing Value Analysis:**
   - Identify and treat outliers.
   - Impute or handle missing values appropriately.

3. **Aggregations:**
   - Aggregate data to the customer level.
   - Calculate averages, totals, and other relevant statistics.

4. **Exploratory Data Analysis (EDA):**
   - Analyze distributions and relationships between variables.
   - Identify key trends and patterns.
   - Visualize data using graphs and charts.

5. **Splitting:**
   - Split the data into training and testing sets (e.g., 70% training, 30% testing).

6. **Machine Learning Algorithm:**
   - Choose appropriate algorithms (e.g., logistic regression, decision trees, random forests, gradient boosting).
   - Train models using the training dataset.
   - Tune hyperparameters for optimal performance.

7. **Evaluation:**
   - Evaluate model performance using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.
   - Validate model using the testing dataset.

8. **Sign-Off / Strategy:**
   - Finalize the best-performing model.
   - Develop strategies based on model insights (e.g., targeted marketing, personalized offers).

### **Post Data Science Execution**

1. **Model Deployment:**
   - Deploy the model into a production environment.
   - Integrate with existing systems (e.g., CRM, marketing platforms).

2. **Model Monitoring:**
   - Continuously monitor model performance.
   - Check for model drift and data drift.
   - Update the model as needed to maintain accuracy.

---

This approach ensures a structured and comprehensive analysis of the customer churn problem, leading to actionable insights and a robust predictive model for the airline company.

### How to Prepare a Machine Learning Algorithm

#### **1. Segregate Study into Supervised & Unsupervised Setup, Classification & Regression Setup**

- **Supervised Learning:**
  - **Classification:** Predict discrete labels (e.g., logistic regression, decision trees, random forests, SVMs, neural networks).
  - **Regression:** Predict continuous values (e.g., linear regression, ridge regression, lasso regression, support vector regression).

- **Unsupervised Learning:**
  - **Clustering:** Group data into clusters (e.g., K-means, hierarchical clustering, DBSCAN).
  - **Dimensionality Reduction:** Reduce feature space (e.g., PCA, t-SNE, LDA).

#### **2. Learn About Assumptions of Algorithms**

- **Linear Regression:**
  - Linearity: The relationship between the input and output is linear.
  - Homoscedasticity: Constant variance of errors.
  - Independence: Observations are independent of each other.
  - Normality: Errors are normally distributed.

- **Logistic Regression:**
  - Linearity of independent variables and log odds.
  - Independence of errors.
  - Absence of multicollinearity.

- **Decision Trees:**
  - No assumptions about data distribution.
  - Can handle both numerical and categorical data.

- **K-Means Clustering:**
  - Assumes spherical clusters.
  - Assumes clusters of similar size.

#### **3. Read Theory About the Algorithm**

- Understand the mathematical foundation and working principle.
- Study the algorithm’s derivation and formulation.
- Learn about the typical use cases and application areas.

#### **4. Go to Scikit-learn Documentation Page**

- **Parameters:**
  - Understand the parameters that can be set before model training.
  - Learn about the default values and their significance.

- **Hyperparameters:**
  - Study the hyperparameters that need tuning for optimal performance.
  - Learn about techniques for hyperparameter tuning (e.g., GridSearchCV, RandomizedSearchCV).

#### **5. Implement the Algorithm on at Least One Dataset**

- Choose a dataset (e.g., from UCI Machine Learning Repository, Kaggle).
- **Example Implementation:**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
# X, y = your data and target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

#### **6. Understand How It Could Be Evaluated**

- **Classification:**
  - Accuracy, precision, recall, F1 score, ROC-AUC, confusion matrix.

- **Regression:**
  - Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared.

- **Clustering:**
  - Silhouette score, Davies-Bouldin index, within-cluster sum of squares.

#### **7. Learn About Limitations & Advantages of the Algorithm**

- **Linear Regression:**
  - **Advantages:** Simple to implement, interpretable coefficients.
  - **Limitations:** Assumes linearity, sensitive to outliers.

- **Logistic Regression:**
  - **Advantages:** Works well with binary classification, interpretable results.
  - **Limitations:** Assumes linear relationship between input and log odds, not suitable for non-linear problems.

- **Decision Trees:**
  - **Advantages:** Easy to understand and interpret, handles both numerical and categorical data.
  - **Limitations:** Prone to overfitting, can be unstable with small changes in data.

- **K-Means Clustering:**
  - **Advantages:** Simple to implement, efficient with large datasets.
  - **Limitations:** Requires specification of number of clusters, assumes spherical clusters.

---

Following these steps ensures a thorough understanding and practical implementation of machine learning algorithms, enhancing both theoretical knowledge and practical skills.


### Data Science Case Study: Airline Customer Churn

#### **Pre-Execution Phase**

**Objective Understanding / Product Understanding / Selecting Success Criteria**

1. **Objective Understanding:**
   - Identify the factors leading to customer churn.
   - Develop a predictive model to forecast potential churn customers.
   - Provide actionable insights to retain customers.

2. **Product Understanding:**
   - Understanding the airline services and customer touchpoints.
   - Analyze customer journey from booking to flight completion.
   - Identify high-value customers and their service usage patterns.

3. **Success Criteria:**
   - Reduction in customer churn rate.
   - Increase in customer retention rate.
   - Improvement in overall customer satisfaction and revenue.

**Response Definition:**
   - Response variable: Churn (1 if the customer has churned, 0 otherwise).

**Data Collection & Data Check:**
   - Collect historical data on customer transactions, demographics, and flight details.
   - Ensure data quality, completeness, and consistency.

#### **Approach**

**Response Variable:**
   - Churn

**Customer Selection Criteria:**
   - Customers who have flown at least 2 years in the last 4 years.
   - Customers who haven’t taken a flight in the past year.
   - Average Flight Value (AFV) > $100.
   - Minimum number of flights: 5.

**Data Variables:**

1. **Demographic Features (Customer Oriented):**
   - Age
   - Gender
   - Destination
   - Departure
   - Class
   - Average amount spent
   - Meal preference (opts for meal or not)
   - Rating
   - Average time between flights
   - Average journey time

2. **Transaction Features:**
   - Mode of payment
   - Reward points usage
   - Transaction time

3. **Flight Features:**
   - Timing of flight (morning/afternoon/evening/night)
   - Weekend or weekday flight
   - Average delay
   - Flight duration

### **Data Science Execution**

1. **Data Cleaning / Standardization:**
   - Handle missing values.
   - Standardize formats (e.g., dates, categorical variables).
   - Remove duplicates and irrelevant data.

2. **Outlier Treatment / Missing Value Analysis:**
   - Identify and treat outliers.
   - Impute or handle missing values appropriately.

3. **Aggregations:**
   - Aggregate data to the customer level.
   - Calculate averages, totals, and other relevant statistics.

4. **Exploratory Data Analysis (EDA):**
   - Analyze distributions and relationships between variables.
   - Identify key trends and patterns.
   - Visualize data using graphs and charts.

5. **Splitting:**
   - Split the data into training and testing sets (e.g., 70% training, 30% testing).

6. **Machine Learning Algorithm:**
   - Choose appropriate algorithms (e.g., logistic regression, decision trees, random forests, gradient boosting).
   - Train models using the training dataset.
   - Tune hyperparameters for optimal performance.

7. **Evaluation:**
   - Evaluate model performance using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.
   - Validate model using the testing dataset.

8. **Sign-Off / Strategy:**
   - Finalize the best-performing model.
   - Develop strategies based on model insights (e.g., targeted marketing, personalized offers).

### **Post Data Science Execution**

1. **Model Deployment:**
   - Deploy the model into a production environment.
   - Integrate with existing systems (e.g., CRM, marketing platforms).

2. **Model Monitoring:**
   - Continuously monitor model performance.
   - Check for model drift and data drift.
   - Update the model as needed to maintain accuracy.

---

This approach ensures a structured and comprehensive analysis of the customer churn problem, leading to actionable insights and a robust predictive model for the airline company.

