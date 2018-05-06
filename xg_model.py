import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, log_loss
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import clean_data


def get_roc(actual, predictions):
    """
    Get the roc curve (and auc score) for the different models
    """
    fig = plt.figure()
    plt.title('ROC Curves')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    colors = ['b', 'g', 'p']

    for model, color in zip(predictions.keys(), colors):
        # Convert preds to just prob of goal
        preds = [pred[1] for pred in predictions[model]]

        false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, preds)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate, true_positive_rate, label=' '.join([model + ':', str(round(roc_auc, 3))]))

    # Add "Random" score
    plt.plot([0, 1], [0, 1], 'r--', label=' '.join(["Random:", str(.5)]))

    plt.legend(title='AUC Score', loc=4)
    fig.savefig("ROC_xG.png")


def fit_gradient_boosting(features_train, labels_train):
    """
    Fit a gradient boosting algorithm and use cross validation to tune the hyperparameters

    :return: classifier
    """
    param_grid = {
        'min_samples_split': [100, 250, 500],
        'max_depth': [3, 4, 5]
    }

    clf = GradientBoostingClassifier(n_estimators=500, learning_rate=.1, random_state=42, verbose=2)

    print("Fitting Gradient Boosting Classifier")

    # Tune hyperparameters
    cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)

    # Fit classifier
    cv_clf.fit(features_train, labels_train)

    print("\nGradient Boosting Classifier:", cv_clf)

    # Save model
    pickle.dump(cv_clf, open("gmb_multiclass_xg.pkl", 'wb'))

    return cv_clf


def fit_random_forest(features_train, labels_train):
    """
    Fit random forest and use cross validation to tune the hyperparameters

    :return: classifier
    """
    param_grid = {
        'min_samples_leaf': [50, 100, 250, 500]
    }

    clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2)

    # Tune hyperparameters
    cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)

    print("Fitting Random Forest")

    cv_clf.fit(features_train, labels_train)
    print("\nRandom Forest Classifier:", cv_clf)

    # Save model
    pickle.dump(cv_clf, open("rf_rebounds_xg.pkl", 'wb'))
    return cv_clf


def fit_logistic(features_train, labels_train):
    """
    Fit the logistic regression and use cross validation to tune the hyperparameters

    :return: classifier
    """
    print("Fitting Logistic")

    param_grid = {
        'C': [.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    }

    clf = LogisticRegression(penalty='l2', solver='sag', random_state=42, max_iter=10000, tol=.01)

    # Tune hyperparameters
    cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)

    # Fit classifier
    cv_clf.fit(features_train, labels_train)

    print("\nLogistic Regression Classifier:", cv_clf)

    # Save model
    pickle.dump(cv_clf, open("logistic_xg.pkl", 'wb'))

    return cv_clf


def xg_model():
    """
    Create and test xg model.
    
    Fit three different models (Refer to those specific functions for more info):
    1. Logistic regression
    2. Gradient Boosting
    3. Random Forest
    """
    data = clean_data.get_data()

    data['Outcome'] = np.where(data['Outcome'] == 0, 0, np.where(data['Outcome'] == 1, 0, np.where(data['Outcome'] == 2, 1, 3)))
    data = data[data['Outcome'] != 3]

    # Convert to lists
    features, labels = clean_data.convert_data(data)

    # Split into training and testing sets -> 80/20
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.2, random_state=42)

    # Fix Data
    features_train, labels_train = np.array(features_train), np.array(labels_train).ravel()

    ###### FIT MODELS
    log_clf = fit_logistic(features_train, labels_train)
    gb_clf = fit_gradient_boosting(features_train, labels_train)
    rf_clf = fit_random_forest(features_train, labels_train)

    #### Testing
    log_preds_probs = log_clf.predict_proba(features_test)
    gb_preds_probs = gb_clf.predict_proba(features_test)
    rf_preds_probs = rf_clf.predict_proba(features_test)

    # Convert test labels to list instead of lists of lists
    flat_test_labels = [label[0] for label in labels_test]

    ### LOG LOSS
    print("\nLog Loss: ")
    print("Logistic Regression: ", log_loss(flat_test_labels, log_preds_probs))
    print("Gradient Boosting: ", log_loss(flat_test_labels, gb_preds_probs))
    print("Random Forest: ", log_loss(flat_test_labels, rf_preds_probs))

    ### ROC
    preds = {
        "Random Forest": rf_preds_probs,
        "Gradient Boosting": gb_preds_probs,
        "Logistic Regression": log_preds_probs
    }
    get_roc(flat_test_labels, preds)


def main():
    xg_model()


if __name__ == '__main__':
    main()

