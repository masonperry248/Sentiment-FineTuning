# Install missing libraries if needed
!pip install -q scikit-learn pandas

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from google.colab import files

# Step 1: Upload your CSV file
print("Upload electronics_reviews.csv")
uploaded = files.upload()  # Make sure to upload the extracted CSV, not the ZIP

# Step 2: Load dataset
df = pd.read_csv("electronics_reviews.csv")

# Step 3: Clean and normalize sentiment column
df['sentiment'] = df['sentiment'].str.lower().str.strip()

# Step 4: Keep only 'positive' and 'negative' sentiments
df = df[df['sentiment'].isin(['positive', 'negative'])]

# Step 5: Encode sentiment labels (positive → 1, negative → 0)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Step 6: View label distribution
print("\nLabel Distribution:")
print(df['label'].value_counts())

# Step 7: Split dataset
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Step 8: Print split sizes
print(f"\nTrain: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

# Step 9: Save to CSV
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Step 10: Download the generated CSVs (optional)
files.download("train.csv")
files.download("val.csv")
files.download("test.csv")
