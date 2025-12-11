import pandas as pd
import re
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# 🔹 Load dataset
file_path = "large_mental_health_dataset.csv"  # Update with actual path
df = pd.read_csv(file_path, encoding="utf-8")

# 🔹 Clean column names
df.columns = df.columns.str.strip().str.lower()

# 🔹 Ensure necessary columns exist
required_columns = {"label", "text", "treatment"}
if not required_columns.issubset(df.columns):
    raise KeyError(f"Missing columns: {required_columns - set(df.columns)}")

# 🔹 Define stopwords
manual_stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "have", "has", "had", "a", "an", "the", "and",
    "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "don", "should", "now"
])

# 🔹 Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join([word for word in text.split() if word not in manual_stopwords])
    return text

# 🔹 Preprocess text
df["cleaned_text"] = df["text"].apply(clean_text)

# 🔹 Encode labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# 🔹 Create dictionary for treatments
treatment_dict = dict(zip(df["label"], df["treatment"]))

# 🔹 Train/test split
X_train, X_test, y_train, y_test = train_test_split(df["cleaned_text"], df["label_encoded"], test_size=0.2, random_state=42)

# 🔹 TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 🔹 Model training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 🔹 Prediction for test
y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# 🔹 Visualization function
def visualize():
    plt.figure(figsize=(10, 5))
    df["label"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Disorder Distribution")
    plt.xticks(rotation=75)
    plt.show()

    if "severity" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="severity", palette="coolwarm")
        plt.title("Severity Levels")
        plt.show()

    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.show()

    df["word_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 5))
    sns.histplot(df["word_count"], bins=30, kde=True, color="purple")
    plt.title("Word Count in Symptoms")
    plt.show()

    all_words = " ".join(df["cleaned_text"]).split()
    common_words = Counter(all_words).most_common(20)
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts), palette="magma")
    plt.title("Top 20 Common Words")
    plt.xticks(rotation=75)
    plt.show()

# 🔹 GUI

def launch_gui():
    def predict_disorder():
        user_input = input_entry.get("1.0", tk.END).strip().lower()
        selected_severity = severity_var.get()

        if not user_input:
            messagebox.showwarning("Input Error", "Please enter your symptoms.")
            return
        if not selected_severity:
            messagebox.showwarning("Input Error", "Please select a severity level.")
            return

        cleaned_input = clean_text(user_input)
        input_vectorized = vectorizer.transform([cleaned_input])
        predicted_label = model.predict(input_vectorized)[0]
        disorder_name = label_encoder.inverse_transform([predicted_label])[0]

        result_label.config(
            text=f"🤔 Prediction: You might have *{disorder_name}*\n🩺 Reported Severity: {selected_severity.capitalize()}"
        )
        result_label.disorder_name = disorder_name
        treatment_button.pack(pady=5)

    def show_treatment():
        disorder = getattr(result_label, 'disorder_name', None)
        if not disorder:
            messagebox.showinfo("Info", "Please predict a disorder first.")
            return
        treatment = treatment_dict.get(disorder, "❌ No treatment info available.")
        result_label.config(text=f"💡 Treatment for {disorder}: {treatment}")

    def show_visuals():
        visualize()

    window = tk.Tk()
    window.title("🧠 Mental Health Assistant")
    window.geometry("700x600")
    window.config(bg="#f0f8ff")

    tk.Label(window, text="Mental Health Assistant", font=("Arial", 18, "bold"), bg="#f0f8ff").pack(pady=10)
    tk.Label(window, text="Describe your symptoms below:", font=("Arial", 14), bg="#f0f8ff").pack()

    input_entry = tk.Text(window, height=6, width=70, font=("Arial", 12))
    input_entry.pack(pady=10)

    tk.Label(window, text="Select Severity Level:", font=("Arial", 13), bg="#f0f8ff").pack()
    severity_var = tk.StringVar()
    severity_dropdown = tk.OptionMenu(window, severity_var, "mild", "moderate", "severe", "critical")
    severity_dropdown.config(font=("Arial", 12))
    severity_dropdown.pack(pady=5)

    tk.Button(window, text="Predict Disorder", command=predict_disorder, bg="#4caf50", fg="white", font=("Arial", 12)).pack(pady=5)

    treatment_button = tk.Button(window, text="Show Treatment", command=show_treatment, bg="#f57c00", fg="white", font=("Arial", 12))
    treatment_button.pack_forget()

    tk.Button(window, text="Show Visualizations", command=show_visuals, bg="#2196f3", fg="white", font=("Arial", 12)).pack(pady=5)

    result_label = tk.Label(window, text="", bg="#f0f8ff", wraplength=600, font=("Arial", 13), fg="#333")
    result_label.pack(pady=20)

    window.mainloop()

# 🔹 Run
if __name__ == "__main__":
    launch_gui()