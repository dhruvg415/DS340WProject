import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("Suicide_Ideation_Dataset(Twitter-based).csv")

# Drop rows with null tweets
df = df.dropna(subset=['Tweet'])

# Encode target labels
df['Suicide'] = df['Suicide'].str.strip().map({
    'Not Suicide post': 0,
    'Potential Suicide post': 1
})

# Clean tweet text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # URLs
    text = re.sub(r'@\w+', '', text)                     # Mentions
    text = re.sub(r'#\w+', '', text)                     # Hashtags
    text = re.sub(r'&\w+;', '', text)                    # HTML entities
    text = re.sub(r'[^a-zA-Z\s]', '', text)              # Punctuation/numbers
    text = text.lower()                                  # Lowercase
    text = re.sub(r'\s+', ' ', text).strip()             # Extra whitespace
    return text

df['cleaned'] = df['Tweet'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned'])
y = df['Suicide'].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Backup indices to retrieve tweet text later for predictions
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df[['Tweet', 'cleaned']], y, test_size=0.2, random_state=42, stratify=y
)

# Reshape for CNN input
X_train_cnn = X_train.toarray().reshape(-1, 5000, 1)
X_test_cnn = X_test.toarray().reshape(-1, 5000, 1)

# CNN Feature Extractor
input_layer = Input(shape=(5000, 1))
x = Conv1D(filters=64, kernel_size=5, activation='relu')(input_layer)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu', name='feature_layer')(x)
output = Dense(1, activation='sigmoid')(x)

cnn_model = Model(inputs=input_layer, outputs=output)
cnn_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = cnn_model.fit(
    X_train_cnn, y_train,
    validation_data=(X_test_cnn, y_test),
    epochs=10,
    batch_size=32,
    class_weight={0: 1, 1: 2},
    verbose=1
)

# Extract CNN features
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('feature_layer').output)

X_train_features = feature_extractor.predict(X_train_cnn)
X_test_features = feature_extractor.predict(X_test_cnn)

# Gradient Boost Classifier
gb_clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    subsample=0.8,
    random_state=42
)

gb_clf.fit(X_train_features, y_train)
y_pred = gb_clf.predict(X_test_features)

# Evaluation
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Show results
print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print("\n📊 Confusion Matrix:\n", cm)
print("\n📝 Classification Report:\n", report)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Suicide', 'Potential Suicide'],
            yticklabels=['Not Suicide', 'Potential Suicide'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot training/validation loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training/validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Create DataFrame for predictions
predictions_df = pd.DataFrame({
    'Tweet': X_test_raw['Tweet'].values,
    'Cleaned_Tweet': X_test_raw['cleaned'].values,
    'True_Label': y_test,
    'Predicted_Label': y_pred
})

# Save to CSV
predictions_df.to_csv("cnn_gradientboost_predictions.csv", index=False)
print("✅ Predictions saved to 'cnn_gradientboost_predictions.csv'")