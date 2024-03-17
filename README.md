# Cars Insurance Claim Prediction

1. We performed Exploratory Data Analysis (EDA) by checking the first few rows, data types, summary statistics, and missing values.
2. We handled missing values by filling them with appropriate methods (median for numeric variables and mode for categorical variables) and dealt with outliers using the IQR method.
3. We converted categorical variables into numerical features using Label Encoding.
4. We selected relevant features based on correlation and domain knowledge for predicting the insurance claim amount.
5. We trained a Linear Regression model, made predictions, and evaluated the model using Mean Squared Error (MSE) and R-squared.

# Fake News Classification

1. We load the train and test datasets and combine the 'title' and 'text' columns into a single 'content' column.
2. We handle missing values by replacing them with empty strings.
3. We use TF-IDF vectorization to convert text data into numerical features.
4. We train a Logistic Regression model using the training dataset.
5. We evaluate the model's performance on the training data using accuracy and classification report.
6. We use the trained model to predict labels for the test dataset.
7. Finally, we save the predictions in a CSV file named 'predictions.csv'.

# Zomato Restaurants Rating
## Explore the Data:
Read the dataset and understand each feature:
url: URL of the restaurant on the Zomato website
address: Address of the restaurant
name: Name of the restaurant
online_order: Whether the restaurant accepts online orders (Yes/No)
book_table: Whether the restaurant allows table booking (Yes/No)
rate: Rating of the restaurant (out of 5)
votes: Number of votes received by the restaurant
phone: Phone number of the restaurant
location: Location of the restaurant
rest_type: Type of the restaurant (e.g., Casual Dining, Cafe, Bar)
dish_liked: Dishes liked by customers
cuisines: Types of cuisines offered by the restaurant
approx_cost(for two people): Approximate cost for two people
reviews_list: Reviews given by customers
menu_item: Menu items offered by the restaurant
listed_in(type): Type of listing (e.g., Buffet, Cafes)
listed_in(city): City in which the restaurant is listed
Explore the dataset info, describe, and find columns with categories and numeric columns.

## Data Cleaning:
Delete redundant columns (e.g., url, address, phone).
Rename columns for better understanding.
Drop duplicates.
Clean individual columns (e.g., convert 'approx_cost(for two people)' to numeric, handle missing values).
Remove NaN values from the dataset.
## Data Visualization:

Explore restaurants delivering online or not.
Explore restaurants allowing table booking or not.
Analyze the relationship between table booking rate and rating.
Identify the best location based on ratings.
Explore the relationship between location and rating.
Analyze the restaurant type.
Analyze the Gaussian restaurant type and rating.
Explore types of services offered by restaurants.
Analyze the relationship between type and rating.
Explore the cost of restaurants.
Analyze the number of restaurants in each location.
Explore the most famous restaurant chains in Bengaluru.

## Machine Learning:
Select appropriate features to build a machine learning model for predicting the rating of a restaurant.
Preprocess the data (e.g., encoding categorical variables, handling missing values).
Split the dataset into training and testing sets.
Choose a suitable machine learning algorithm (e.g., regression).
Train the model on the training set.
