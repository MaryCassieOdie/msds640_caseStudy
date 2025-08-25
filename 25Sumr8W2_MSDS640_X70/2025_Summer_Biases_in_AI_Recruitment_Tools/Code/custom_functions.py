import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "KNN": KNeighborsClassifier()
    }

#EDA plots
def plot_eda(df):

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

# Remove exclusions
def remove_texts(cleaned_text, exclusions):

    if(len(exclusions) > 0):
        # Tokenize
        cleaned_text = cleaned_text.apply(lambda x: x.split())
        cleaned_text = cleaned_text.apply(lambda x: [item for item in x if item not in exclusions])
        # Join tokens back into a single string
        cleaned_text = cleaned_text.apply(lambda x: ' '.join(x))
    
    return cleaned_text

# Fairness metrics
def fairness_metric(X_train, y_train, X_test, y_test, s_train, s_test):

    tfidf = TfidfVectorizer(stop_words="english", max_features=None)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate
    }

    frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=s_test
    )

    print(frame.by_group)

    # Apply fairness mitigator (ExponentiatedGradient + DemographicParity)
    # Note: Requires dense arrays
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    constraint = DemographicParity()
    base_estimator = LogisticRegression(max_iter=1000)
    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint
    )
    mitigator.fit(X_train_dense, y_train, sensitive_features=s_train)
    y_pred_mitigated = mitigator.predict(X_test_dense)

    # Fairness evaluation (after mitigation)
    frame_mitigated = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred_mitigated,
        sensitive_features=s_test
    )

    print("\n\nAfter Mitigation:\n")
    print(frame_mitigated.by_group)

# Supervised machine learning - classifiers with ROC-AUC graph
def evaluate_classifiers_with_roc_auc_graph(X_train, y_train, X_test, y_test):

    results = []
    plt.figure(figsize=(8, 6))
    # Binarize labels for multi-class AUC
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        model_filename = name + "_model.joblib"
        joblib.dump(clf, model_filename)
        
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1-score": f1_score(y_test, y_pred, average='weighted')
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