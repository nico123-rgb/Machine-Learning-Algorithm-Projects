

# Import libraries - these are toolboxes that help us work with data and machine learning
# numpy helps us work with numbers and math
# pandas helps us work with tables of data  
# sklearn provides machine learning tools for prediction and evaluation
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the car data from a CSV file into a table
# Make a copy so we don't accidentally damage the original data
df = pd.read_csv("Automobile.csv")
df1 = df.copy()

# Remove columns we don't need for prediction
# Car names don't help predict origin
df1.drop(["name"], axis=1, inplace=True)
# Not using engine displacement
df1.drop(["displacement"], axis=1, inplace=True)
# Year doesn't matter for this prediction
df1.drop(["model_year"], axis=1, inplace=True)

# Set what we want to predict
# The origin column tells us where car is from (USA, Europe, Japan)
target_col = "origin"

# Split the data into two parts
# X = all the car features (mpg, weight, horsepower, etc.)
X = df1.drop(columns=[target_col]).copy()
# y = just the origin (what we want to predict)
y = df1[target_col].copy()

# Fill in any missing numbers
# If a car is missing data, use the middle value (median) of that column
X = X.fillna(X.median(numeric_only=True))

# Clean up the origin labels
# Make them all lowercase and remove extra spaces
y = y.astype(str).str.strip().str.lower().to_numpy()

# Create a dictionary to display nice names when we print results
display_name_map = {
    "usa": "USA",
    "europe": "Europe",
    "japan": "Japan",
}
# Get unique origin values and sort them
classes_sorted = np.sort(np.unique(y))
# Create nice display names for each origin
class_names = [display_name_map.get(c, c.title()) for c in classes_sorted]

# Show an overview of what data we have
print("=" * 60)
print("AUTOMOBILE DATASET OVERVIEW")
print("=" * 60)
# How many cars total
print(f"Number of samples:  {X.shape[0]}")
# How many characteristics per car
print(f"Number of features: {X.shape[1]}")
# What characteristics we're looking at
print(f"Feature names:      {list(X.columns)}")
# The three possible origins
print(f"Class names:        {class_names}")
print(f"\nClass distribution:")
# Count how many cars from each country
for c, name in zip(classes_sorted, class_names):
    count = np.sum(y == c)
    print(f"  {name}: {count} samples ({count/len(y)*100:.1f}%)")

# Show example cars so we can see what the data looks like
print(f"\nFirst 5 rows of the dataset:")
print(X.head())

# Show statistics (average, min, max, etc.) for each feature
print(f"\nFeature Statistics:")
print(X.describe().round(2))


# Split data into training (70%) and testing (30%) groups
# Training = cars we'll teach the computer with
# Testing = cars we'll quiz the computer on later
# Keep same proportion of USA/Europe/Japan in both groups
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# Show how many cars ended up in each group
print("\n" + "=" * 60)
print("TRAIN/TEST SPLIT")
print("=" * 60)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")


# Standardize (scale) all the numbers so they're comparable
# This transforms everything so the average is 0 and standard deviation is 1
# Create the scaling tool
scaler = StandardScaler()
# Learn scale from training data and apply it
X_train_scaled = scaler.fit_transform(X_train)
# Apply same scale to testing data
X_test_scaled = scaler.transform(X_test)

# Show an example of how scaling works - compare before and after
first_feature = X.columns[0]
print(f"\nBefore scaling - {first_feature} mean: {X_train[first_feature].mean():.2f}")
print(f"After scaling  - {first_feature} mean: {X_train_scaled[:, 0].mean():.4f}")
print("(Should be approximately 0 after standardization)")


# Create the prediction model (logistic regression)
# This is the "brain" that will learn to predict car origins
model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    C=1.0,
    random_state=42,
)


# Train the model - teach it patterns from the training data
# It learns things like "heavy cars with low mpg tend to be American"
model.fit(X_train_scaled, y_train)

# Show training results
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)
# How many tries it took to learn
print(f"Number of iterations used: {model.n_iter_[0]}")
# Shape of learned weights
print(f"Coefficients shape: {model.coef_.shape}")
# One set of weights per origin
print("(One row of coefficients per class)")


# Use the trained model to make predictions on the test cars
# These are cars it's never seen before
# Get predicted origin for each car
y_pred = model.predict(X_test_scaled)
# Get probability scores (like 70% USA, 20% Europe, 10% Japan)
y_proba = model.predict_proba(X_test_scaled)

# Show example predictions for the first 5 test cars
print("\n" + "=" * 60)
print("PREDICTIONS (First 5 Test Samples)")
print("=" * 60)
for i in range(min(5, len(y_test))):
    # Create dictionary of probabilities for each origin
    proba_dict = {
        display_name_map.get(cls, str(cls).title()): f"{prob:.3f}"
        for cls, prob in zip(model.classes_, y_proba[i])
    }
    # Get nice names for actual and predicted origins
    actual_name = display_name_map.get(y_test[i], str(y_test[i]).title())
    pred_name = display_name_map.get(y_pred[i], str(y_pred[i]).title())
    # Print what it actually was, what we predicted, and all probability scores
    print(
        f"  Sample {i+1}: Actual={actual_name}, "
        f"Predicted={pred_name}, Probabilities={proba_dict}"
    )





# Calculate accuracy - what percentage of predictions were correct
accuracy = accuracy_score(y_test, y_pred)
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")


# Create confusion matrix - shows where the model made mistakes
# Rows = actual origin, Columns = predicted origin
cm = confusion_matrix(y_test, y_pred, labels=classes_sorted)
print(f"\nConfusion Matrix:")
print(pd.DataFrame(cm, index=class_names, columns=class_names))


# Print detailed report showing precision, recall, and f1-score for each origin
# Precision = when model says "USA", how often is it right?
# Recall = of all actual USA cars, how many did we catch?
print(f"\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=classes_sorted,
        target_names=class_names,
    )
)


# Show which car features are most important for predicting each origin
print("=" * 60)
print("FEATURE IMPORTANCE (Top 3 per class)")
print("=" * 60)

# Create a table of coefficients (importance weights) for each feature
# These are the learned weights
coef_df = pd.DataFrame(
    model.coef_,
    columns=X.columns,
    index=[display_name_map.get(c, str(c).title()) for c in model.classes_],
)

# For each origin, find and display the top 3 most important features
for class_name in coef_df.index:
    # Get 3 features with biggest impact
    top_features = coef_df.loc[class_name].abs().nlargest(3)
    print(f"\n  {class_name}:")
    for feat in top_features.index:
        actual_val = coef_df.loc[class_name, feat]
        # Does this feature help or hurt prediction?
        direction = "increases" if actual_val > 0 else "decreases"
        print(f"    {feat}: {actual_val:.3f} ({direction} probability)")

# Print completion message
print("\n" + "=" * 60)
print("DONE! Automobile origin classification complete.")
print("=" * 60)



import matplotlib
try:
    # Use TkAgg backend for displaying graphs
    matplotlib.use("TkAgg")
except Exception:
    # If it fails, use default backend
    pass
# Library for creating graphs
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Create simple made-up data for demonstration
# X values
X = np.array([2, 3, 4, 5, 6, 7, 7, 8, 9, 11, 12]).reshape(-1, 1)
# Y values
y_raw = np.array([18, 16, 15, 17, 20, 23, 25, 28, 31, 30, 29])

# Convert continuous values to binary classes (0 or 1)
# Anything >= 24 becomes class 1, anything < 24 becomes class 0
threshold = 24
# Convert to 0s and 1s
y = (y_raw >= threshold).astype(int)

# Create and train a simple logistic regression model for demonstration
log_model = LogisticRegression()
# Learn from the made-up data
log_model.fit(X, y)

# Create smooth prediction curve for the graph
# Create 200 evenly spaced X values
X_pred = np.linspace(2, 12, 200).reshape(-1, 1)
# Get probability of being class 1 at each point
y_prob = log_model.predict_proba(X_pred)[:, 1]
# Get predicted class at each point
y_class = log_model.predict(X_pred)

# Calculate decision boundary (the X value where probability = 50%)
# This is where the model switches from predicting class 0 to class 1
decision_boundary = -log_model.intercept_[0] / log_model.coef_[0][0]


# Create the graph
# Make figure 8 inches wide, 5 inches tall
plt.figure(figsize=(8, 5))
# Plot class 0 points in blue
plt.scatter(X[y == 0], y[y == 0], color="blue", label="Class 0")
# Plot class 1 points in green
plt.scatter(X[y == 1], y[y == 1], color="green", label="Class 1")
# Draw probability curve
plt.plot(X_pred, y_prob, color="red", linewidth=2, label="Logistic curve P(class=1)")
# Draw vertical line at decision point
plt.axvline(decision_boundary, color="purple", linestyle="--", label=f"Decision boundary = {decision_boundary:.2f}")

# Add labels and format the graph
# Label x-axis
plt.xlabel("X")
# Label y-axis
plt.ylabel("Probability / Class")
# Add title
plt.title("Logistic Regression")
# Set y-axis limits
plt.ylim(-0.1, 1.1)
# Show legend explaining what each color means
plt.legend()
# Add light grid lines
plt.grid(alpha=0.3)
# Adjust spacing so everything fits nicely
plt.tight_layout()
# Display the graph window
plt.show(block=True)

#The purpose of this Multinomial Logistic Regression model is to classfiy cars into three orgion countries (USA, Europe, Japan) based on their characteristics (mpg, weight, horsepower, etc.). This can be used to understand waht cars from each country tend to be like and help cunsomers make informed decisions as well as help manufacturers make targeted improvemnets. 
