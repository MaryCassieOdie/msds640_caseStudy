import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import numpy as np
import joblib
import re

classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

def display_unique_category_values(df):
    df['Gender'].value_counts()
    df['Race'].value_counts()
    df['Ethnicity'].value_counts()
    df['Job Roles'].value_counts()
    df['Best Match'].value_counts()

#plot
def plot_eda(df):
    gender_counts = df.groupby(['Best Match', 'Race']).size().unstack(fill_value=0)
    gender_counts.plot(kind='bar', stacked=True, figsize=(6, 4))
    plt.title('Success and Failure Count Based on Race')
    plt.xlabel('Success(1) or Failure(0)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(title='Race')
    plt.tight_layout()
    plt.show()

    df['Job Roles'].value_counts().plot(kind='bar', title='Frequency of Job Roles', figsize=(10, 8))
    plt.xlabel('Job Roles')
    plt.ylabel('Count')
    plt.show()

    gender_counts = df.groupby(['Job Roles', 'Gender']).size().unstack(fill_value=0)
    gender_counts.plot(kind='bar', stacked=True, figsize=(14, 10))
    plt.title('Gender Wise Job Role')
    plt.xlabel('Job Role')
    plt.ylabel('Count')
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.show()

    gender_counts = df.groupby(['Job Roles', 'Race']).size().unstack(fill_value=0)
    gender_counts.plot(kind='bar', stacked=True, figsize=(26, 10))
    plt.title('Race Wise Job Role')
    plt.xlabel('Job Role')
    plt.ylabel('Count')
    plt.legend(title='Race')
    plt.tight_layout()
    plt.show()

    gender_counts = df.groupby(['Job Roles', 'Ethnicity']).size().unstack(fill_value=0)
    gender_counts.plot(kind='bar', stacked=True, figsize=(26, 8))
    plt.title('Ethnicity Wise Job Role')
    plt.xlabel('Job Role')
    plt.ylabel('Count')
    plt.legend(title='Ethnicity')
    plt.tight_layout()
    plt.show()

# Clean
def clean_resume(text):
    #  # remove URLs
    #  # remove mentions
    


    # Remove RT and cc
    text = re.sub('RT|cc', ' ', text)
    # Remove @mentions
    text = re.sub('@\S+', '  ', text)
    # Remove URLs
    text = re.sub('http\S+\s*', ' ', text)
    # Remove hashtags
    text = re.sub('#\S+', '', text)
    # Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    # Remove numbers
    text = re.sub(r'[0-9]+', '', text)
    # Remove extra spaces
    text = re.sub('\s+', ' ', text)
    # Remove leading and trailing spaces
    text = re.sub(r"^\s+|\s+$", "", text)
    # Replace multiple internal spaces with a single space
    text = re.sub(r"\s+", " ", text)

    #text = re.sub(r"\s+", " ", text.strip())

    return text

# Pre process
def pre_process(cleaned_text, stopwords, exclusions, stemmer):

    # Convert to lowercase
    cleaned_text = cleaned_text.str.lower()
    
    # Tokenize
    cleaned_text = cleaned_text.apply(lambda x: x.split())
    
    # Remove stopwords
    cleaned_text = cleaned_text.apply(lambda x: [item for item in x if item not in stopwords])
    
    # Remove Exclusions
    if(len(exclusions) > 0):
        cleaned_text = cleaned_text.apply(lambda x: [item for item in x if item not in exclusions])

    # Apply stemming
    cleaned_text = cleaned_text.apply(lambda x: [stemmer.stem(i) for i in x])
    
    # Join tokens back into a single string
    cleaned_text = cleaned_text.apply(lambda x: ' '.join(x))
    
    return cleaned_text

# Pre process
def remove_texts(cleaned_text, exclusions):

    if(len(exclusions) > 0):
        # Tokenize
        cleaned_text = cleaned_text.apply(lambda x: x.split())
        cleaned_text = cleaned_text.apply(lambda x: [item for item in x if item not in exclusions])
        # Join tokens back into a single string
        cleaned_text = cleaned_text.apply(lambda x: ' '.join(x))
    
    return cleaned_text

# Supervised machine learning - classifiers
def evaluate_classifiers(X_train, y_train, X_test, y_test):
    
    results = []

    plt.figure(figsize=(8, 6))
    # Binarize labels for multi-class AUC
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        model_filename = "model/" + name + "_model.joblib"
        joblib.dump(clf, model_filename)
        
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1-score": f1_score(y_test, y_pred, average='weighted'),
            "Training & test": str('{:.3f}'.format(clf.score(X_train, y_train))) + " & " + str('{:.3f}'.format(clf.score(X_test, y_test)))
        })
    
        # ROC Curve & AUC
        try:
            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X_test)
            else:
                y_score = clf.decision_function(X_test)
            
            if n_classes > 2:
                auc = roc_auc_score(y_test_bin, y_score, average='weighted', multi_class='ovr')
                fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            else:
                auc = roc_auc_score(y_test, y_score[:, 1])
                fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
            
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
        
        except Exception as e:
            print(f"[Skipped ROC for {name}] {e}")
    
    # Finalize plot
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC/AUC Curves")
    plt.legend()
    plt.show()

    return pd.DataFrame(results)

# Live data evaluation for supervised learning- classifier
def live_data_evaluation(live_word_features, y_live_target):
    results = []
    
    for name, clf in classifiers.items():

        model_filename = "model/" + name + "_model.joblib"

        loaded_model = joblib.load(model_filename)
        y_live_pred = loaded_model.predict(live_word_features)

        loaded_model.score(live_word_features, y_live_target)
        
        results.append({
            "Model": name,
            "Accuracy on live data set": accuracy_score(y_live_target, y_live_pred),
            #"Score": loaded_model.score(live_word_features, y_live_target)
        })
    
    return pd.DataFrame(results)