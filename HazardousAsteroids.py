import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')
def load_asteroid_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print("File not found: " + file_path)
        return None
    except pd.errors.EmptyDataError:
        print("File is empty: " + file_path)
        return None
    except pd.errors.ParserError:
        print("Error parsing file: " + file_path)
        return None

def process_and_classify_asteroid_data(data):
    required_columns = [
        'full_name', 'a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc', 'H'
    ]
    for col in required_columns:
        if (col not in data.columns) or (data[col].isnull().any()):
            print("Missing or empty required column: " + col)
            return None

    asteroid_info = []
    for _, asteroid in data.iterrows():
        full_name = asteroid['full_name']
        a = asteroid['a']  # Semi-major axis
        e = asteroid['e']  # Eccentricity
        i = asteroid['i']
        om = asteroid['om']  # Longitude of the ascending node
        w = asteroid['w']  # Argument of perihelion
        q = asteroid['q']  # Perihelion distance
        ad = asteroid['ad']  # Aphelion distance
        per_y = asteroid['per_y']
        data_arc = asteroid['data_arc']
        H = asteroid['H']  # Absolute magnitude
        # Calculate Earth Minimum Orbit Intersection Distance (MOID)
        MOID = q * (1 - e)
        # Determine if the asteroid is a Potentially Hazardous Asteroid (PHA)
        is_pha = 1 if (MOID <= 0.05) and (H <= 22.0) else 0
        info = {
            'full_name': full_name,
            'a': a,
            'e': e,
            'i': i,
            'om': om,
            'w': w,
            'q': q,
            'ad': ad,
            'per_y': per_y,
            'data_arc': data_arc,
            'H': H,
            'MOID': MOID,
            'is_pha': is_pha,
        }
        asteroid_info.append(info)
    df = pd.DataFrame(asteroid_info)
    return df

def train_svm(X_train, y_train, X_test, y_test):
    # Define parameter grid for SVM
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf'],
    }

    # Perform grid search with cross-validation
    svm_grid_search = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5)
    svm_grid_search.fit(X_train, y_train)
    svm_best_model = svm_grid_search.best_estimator_
    # Make predictions
    svm_pred = svm_best_model.predict(X_test)
    # Calculate accuracy
    svm_accuracy = accuracy_score(y_test, svm_pred)
    # Generate confusion matrix and classification report
    svm_conf_matrix = confusion_matrix(y_test, svm_pred)
    svm_classification_rep = classification_report(y_test, svm_pred, zero_division=0)
    # Print accuracy, confusion matrix, and classification report
    print("\u001b[32mBest SVM Model accuracy: \u001b[0m" + str(svm_accuracy))
    print("Confusion Matrix:")
    print(svm_conf_matrix)
    print("Classification Report:")
    print(svm_classification_rep)
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(svm_conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
                xticklabels=['Not PHA', 'PHA'], yticklabels=['Not PHA', 'PHA'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - SVM Model')
    plt.show()
    return svm_best_model, svm_accuracy

def train_logistic_regression(X_train, y_train, X_test, y_test):
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    log_reg_pred = log_reg.predict(X_test)
    log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
    # Metrics
    log_reg_conf_matrix = confusion_matrix(y_test, log_reg_pred)
    log_reg_classification_rep = classification_report(y_test, log_reg_pred, zero_division=1)
    print("\u001b[32mLogistic Regression accuracy: \u001b[0m" + str(log_reg_accuracy))
    print("Confusion Matrix:")
    print(log_reg_conf_matrix)
    print("Classification Report:")
    print(log_reg_classification_rep)
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(log_reg_conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
                xticklabels=['Not PHA', 'PHA'], yticklabels=['Not PHA', 'PHA'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Logistic Regression Model')
    plt.show()
    return log_reg, log_reg_accuracy

def train_decision_tree(X_train, y_train, X_test, y_test):
    dt_param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
    }
    dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=5)
    dt_grid_search.fit(X_train, y_train)
    dt_best_model = dt_grid_search.best_estimator_
    # Predictions
    dt_pred = dt_best_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    # Metrics
    dt_conf_matrix = confusion_matrix(y_test, dt_pred)
    dt_classification_rep = classification_report(y_test, dt_pred, zero_division=1)
    print("\u001b[32mBest Decision Tree Model accuracy: \u001b[0m" + str(dt_accuracy))
    print("Confusion Matrix:")
    print(dt_conf_matrix)
    print("Classification Report:")
    print(dt_classification_rep)
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(dt_conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
                xticklabels=['Not PHA', 'PHA'], yticklabels=['Not PHA', 'PHA'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Decision Tree Model')
    plt.show()
    return dt_best_model, dt_accuracy

def train_naive_bayes(X_train, y_train, X_test, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    # Metrics
    nb_conf_matrix = confusion_matrix(y_test, nb_pred)
    nb_classification_rep = classification_report(y_test, nb_pred, zero_division=1)
    print("\u001b[32mNaive Bayes accuracy: \u001b[0m" + str(nb_accuracy))
    print("Confusion Matrix:")
    print(nb_conf_matrix)
    print("Classification Report:")
    print(nb_classification_rep)
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(nb_conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
                xticklabels=['Not PHA', 'PHA'], yticklabels=['Not PHA', 'PHA'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Naive Bayes Model')
    plt.show()
    return nb, nb_accuracy

def train_knn(X_train, y_train, X_test, y_test):
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }
    knn_grid_search = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5)
    knn_grid_search.fit(X_train, y_train)
    knn_best_model = knn_grid_search.best_estimator_
    # Predictions
    knn_pred = knn_best_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    # Metrics
    knn_conf_matrix = confusion_matrix(y_test, knn_pred)
    knn_classification_rep = classification_report(y_test, knn_pred)
    print("\u001b[32mBest k-Nearest Neighbors Model accuracy: \u001b[0m" + str(knn_accuracy))
    print("Confusion Matrix:")
    print(knn_conf_matrix)
    print("Classification Report:")
    print(knn_classification_rep)
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(knn_conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
                xticklabels=['Not PHA', 'PHA'], yticklabels=['Not PHA', 'PHA'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - k-Nearest Neighbors Model')
    plt.show()
    return knn_best_model, knn_accuracy

def train_model(data):
    X = data[['a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc', 'H']]
    y = data['is_pha']
    # Using StratifiedShuffleSplit to create train/test splits
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Dictionary to store trained models and their accuracies
    models = {}
    # Train and store each model
    models['SVM'] = train_svm(X_train, y_train, X_test, y_test)
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train, X_test, y_test)
    models['Decision Tree'] = train_decision_tree(X_train, y_train, X_test, y_test)
    models['k-Nearest Neighbors'] = train_knn(X_train, y_train, X_test, y_test)
    models['Naive Bayes'] = train_naive_bayes(X_train, y_train, X_test, y_test)
    return models

def visualize_asteroids(data):
    fig = px.scatter_3d(
        data, x='a', y='e', z='i',
        color='is_pha',
        title='Potentially Hazardous Asteroids',
        labels={
            'a': 'Semi-Major Axis (AU)',
            'e': 'Eccentricity',
            'i': 'Inclination (Degrees)'
        },
        size='H',
        hover_name='full_name',
        template='plotly_dark'
    )
    fig.update_layout(scene=dict(
        xaxis=dict(range=[min(data['a']), max(data['a'])]),
        yaxis=dict(range=[min(data['e']), max(data['e'])]),
        zaxis=dict(range=[min(data['i']), max(data['i'])]),
    ))
    fig.show()

def search_asteroids(data, search_name):
    results = data[data['full_name'].str.contains(search_name, case=False, na=False)]
    return results

def predict_hazard_status(model, features):
    """
    Predicts the hazard status using the given model and features.
    """
    features = [features]  # Convert features to a 2D array as expected by model.predict()
    prediction = model.predict(features)
    hazard_status = 'Potentially Hazardous' if prediction == 1 else 'Not Potentially Hazardous'
    return hazard_status
import plotly.graph_objects as go
def run_user_interface(data, models):
    search_name = input("Enter asteroid name to search: ")
    search_results = search_asteroids(data, search_name)

    asteroid = data[data['full_name'].str.contains(search_name, case=False, na=False)].iloc[0]
    # Create 3D plot
    fig = go.Figure()
    # Add Earth's orbit
    earth_orbit = go.Scatter3d(
        x=np.cos(np.linspace(0, 2*np.pi, 100)),
        y=np.sin(np.linspace(0, 2*np.pi, 100)),
        z=[0]*100,
        mode='lines',
        name='Earth Orbit'
    )
    fig.add_trace(earth_orbit)
    # Add asteroid's orbit
    a = asteroid['a']
    e = asteroid['e']
    i = np.radians(asteroid['i'])
    omega = np.radians(asteroid['om'])
    w = np.radians(asteroid['w'])
    theta = np.linspace(0, 2*np.pi, 100)
    r = (a * (1 - e**2)) / (1 + e * np.cos(theta))
    x_orbit = r * (np.cos(omega) * np.cos(w + theta) - np.sin(omega) * np.sin(w + theta) * np.cos(i))
    y_orbit = r * (np.sin(omega) * np.cos(w + theta) + np.cos(omega) * np.sin(w + theta) * np.cos(i))
    z_orbit = r * (np.sin(i) * np.sin(w + theta))
    asteroid_orbit = go.Scatter3d(
        x=x_orbit,
        y=y_orbit,
        z=z_orbit,
        mode='lines',
        name=asteroid['full_name']
    )
    fig.add_trace(asteroid_orbit)
    fig.update_layout(
        title=f'Orbit of {asteroid["full_name"]}',
        scene=dict(
            xaxis_title='X (AU)',
            yaxis_title='Y (AU)',
            zaxis_title='Z (AU)'
        )
    )
    fig.show()

    if not search_results.empty:
        for _, row in search_results.iterrows():
            print("\nAsteroid Name: " +row['full_name'])
            print("Semi-Major Axis (AU): " +str(row['a']))
            print("Eccentricity: " +str(row['e']))
            print("Inclination (Degrees): " +str(row['i']))
            print("Orbital Period (Years): " +str(row['per_y']))
            print("Absolute Magnitude: " + str(row['H']))
            print("Data Arc (Days): " + str(row['data_arc']))
            print("Potentially Hazardous: " + ("Yes" if row['is_pha'] == 1 else "No"))
            # Extract the features needed for prediction
            features = [row['a'], row['e'], row['i'], row['om'], row['w'], row['q'], row['ad'], row['per_y'], row['data_arc'], row['is_pha']]
            # Debug print to inspect the models dictionary
            print("Models Dictionary:", models)

            # Predict using each model and display predictions
            for model_name in models:
                model_info = models[model_name]  # Access the tuple
                print(f"Model Info for {model_name}: {model_info}")  # Debug print to inspect model_info
                # Ensure model_info is a tuple with at least two elements (model, accuracy)
                if isinstance(model_info, tuple) and len(model_info) >= 2:
                    model = model_info[0]
                    accuracy = model_info[1]
                    hazard_status = predict_hazard_status(model, features)
                    print(f"{model_name} Prediction: {hazard_status} (Accuracy: {accuracy})")
                else:
                    print(f"Error: Model info for {model_name} is not in the expected format (model, accuracy).")

def main():
    # Load asteroid data
    asteroid_data = load_asteroid_data_from_csv("/content/drive/MyDrive/Colab Notebooks/data project ai.csv")
    if asteroid_data is None:
        return
    # Process and classify asteroid data
    processed_data = process_and_classify_asteroid_data(asteroid_data)
    if processed_data is None:
        return
    # Train models
    models = train_model(processed_data)
    # Visualize asteroid data
    visualize_asteroids(processed_data)
    # User interface
    run_user_interface(processed_data, models)

if __name__ == "__main__":
    main()
