
# Import libraries for math, graphing, and data handling
import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

# Load the spam email dataset
df=pd.read_csv('spam_email_dataset.csv')

# Select features for prediction and target variable
feature_cols = ["Attachments", "Link Count", "Word Count", "Uppercase Count", "Exclamation Count", "Question Count", "Dollar Count", "Punctuation Count", "HTML Tags Count"]
x = df[feature_cols].values
y = df["Spam Indicator"].values

# Split data into training (75%) and testing (25%) sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# Normalize data to improve model performance
from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)


# Train logistic regression classifier on the data
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0, max_iter=1000)
classifier.fit(x_train,y_train)

# Make predictions on test set
y_pred=classifier.predict(x_test)


# Calculate model accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy}")

# Calculate precision, recall, and F1-score
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Create visualization with Link Count feature
plot_feature = "Link Count"
plot_idx = feature_cols.index(plot_feature)

y_curve_x = nm.linspace(df[plot_feature].min(), df[plot_feature].max(), 200)
X_curve_df = pd.DataFrame(
    nm.tile(df[feature_cols].median().to_numpy(), (200, 1)),
    columns=feature_cols,
)
X_curve_df[plot_feature] = y_curve_x

# Get probability predictions for the curve
X_curve_scaled = st_x.transform(X_curve_df)
y_curve_prob = classifier.predict_proba(X_curve_scaled)[:, 1]

# Calculate decision boundary where model switches prediction
w = classifier.coef_[0]
b = classifier.intercept_[0]
z_fixed = b + nm.dot(X_curve_scaled[0], w) - w[plot_idx] * X_curve_scaled[0, plot_idx]
if abs(w[plot_idx]) > 1e-12:
    boundary_scaled = -z_fixed / w[plot_idx]
    boundary_raw = boundary_scaled * st_x.scale_[plot_idx] + st_x.mean_[plot_idx]
else:
    boundary_raw = nm.nan

# Create scatter plot with logistic curve and decision boundary
plt.figure(figsize=(8, 5))
plt.scatter(df[y == 0][plot_feature], y[y == 0], color="blue", alpha=0.55, label="Not Spam (0)")
plt.scatter(df[y == 1][plot_feature], y[y == 1], color="green", alpha=0.55, label="Spam (1)")
plt.plot(y_curve_x, y_curve_prob, color="red", linewidth=2, label=f"Logistic curve P(spam=1) vs {plot_feature}")

if not nm.isnan(boundary_raw):
    plt.axvline(boundary_raw, color="purple", linestyle="--", label=f"Decision boundary = {boundary_raw:.2f}")

plt.xlabel(plot_feature)
plt.ylabel("Probability / Class")
plt.title("Binary Logistic Regression for Email Spam")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Print model parameters and boundary
print("Intercept:", classifier.intercept_[0])
print("Coefficient for Link Count:", classifier.coef_[0][plot_idx])
if not nm.isnan(boundary_raw):
    print("Decision boundary (Link Count):", boundary_raw)

# Interactive prediction loop
while True:
    try:
        attachments = input("Enter Attachments: ")
        attachments = int(attachments)
        link_count = int(input("Enter Link Count: "))
        word_count = int(input("Enter Word Count: "))
        uppercase_count = int(input("Enter Uppercase Count: "))
        exclamation_count = int(input("Enter Exclamation Count: "))
        question_count = int(input("Enter Question Count: "))
        dollar_count = int(input("Enter Dollar Count: "))
        punctuation_count = int(input("Enter Punctuation Count: "))
        html_tags_count = int(input("Enter HTML Tags Count: "))

        # Combine features into array
        new_data_point = nm.array([[attachments, link_count, word_count, uppercase_count, exclamation_count, question_count, dollar_count, punctuation_count, html_tags_count]])
        
        # Scale data and make prediction
        new_data_point = st_x.transform(new_data_point)
        prediction = classifier.predict(new_data_point)

        if prediction[0] == 0:
          print("Prediction: Not Spam")
        else:
          print("Prediction: Spam")

        break
    except ValueError:
        print("Invalid input. Please enter valid numeric values for all email features.")
    except Exception as e:
        print(f"An error occurred: {e}")





















