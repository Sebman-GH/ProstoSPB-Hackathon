import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

class HandleDataset():
    def __init__(self):
        pass

    @staticmethod
    def get_df(path):
        return pd.read_csv(path)

    @staticmethod
    def get_df_with_molecular_descriptors(df):
        @staticmethod
        def _get_molecular_descriptors_from_smiles(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [None] * len(descriptor_names)
            descriptors = descriptor_calculator.CalcDescriptors(mol)
            return descriptors

        descriptor_names = [desc[0] for desc in Chem.Descriptors._descList]
        descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        descriptors_df = df['SMILES'].apply(_get_molecular_descriptors_from_smiles).apply(pd.Series)
        descriptors_df.columns = descriptor_names
        return pd.concat([df, descriptors_df], axis=1)

    @staticmethod
    def filter_df_from_outliers(df):
        q3 = df.quantile(0.75)
        return df.loc[:, q3 > 0]
    


class ModelClassification():
    def __init__(self):
        pass

    @staticmethod
    def train_with_XGBClassifier(df):
        X = df.drop('Class', axis=1)
        y = df['Class']

        X = X.select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        xgb_model = xgb.XGBClassifier(eval_metric='logloss')

        test_accuracies = []

        for i in range(1, X_train.shape[1] + 1):
            xgb_model.fit(X_train.iloc[:, :i], y_train, eval_set=[(X_test.iloc[:, :i], y_test)], 
                        verbose=False)
            
            y_test_pred = xgb_model.predict(X_test.iloc[:, :i])
            
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            test_accuracies.append(test_accuracy)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, X_train.shape[1] + 1), test_accuracies, label='Test Accuracy', color='orange')
        plt.title('Accuracy over Number of Features')
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def get_importance_features_by_xgb(df):
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X = X.select_dtypes(include=[np.number])

        xgb_model = xgb.XGBClassifier(eval_metric='logloss')
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        importance = xgb_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        print("Важность признаков:")
        print(importance_df)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Важность признаков')
        plt.gca().invert_yaxis()
        plt.show()

        evals = [(X_train, y_train), (X_test, y_test)]
        history = xgb_model.fit(X_train, y_train, eval_set=evals, 
                                verbose=True)

        test_accuracy_history = history.evals_result()['validation_1']['logloss']

        plt.figure(figsize=(10, 6))
        plt.plot(test_accuracy_history, label='Test Logloss', color='orange')
        plt.title('Logloss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Logloss')
        plt.legend()
        plt.grid()
        plt.show()
        return importance_df

    def get_most_important_features(df):
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X = X.select_dtypes(include=[np.number])
        xgb_model = xgb.XGBClassifier(eval_metric='logloss')
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        importance = xgb_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        features_to_keep = importance_df[importance_df['Importance'] > 0.01]['Feature']
        return df[['Class'] + features_to_keep.tolist()]
    
    def train_melanin_with_importance_features(df):
        X = df.drop('Class', axis=1)
        y = df['Class']

        X = X.select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        xgb_model = xgb.XGBClassifier(eval_metric='logloss')
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        importance = xgb_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        print("Важность признаков:")
        print(importance_df)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Важность признаков')
        plt.gca().invert_yaxis()
        plt.show()

    def train_model_with_best_accuracy(df):
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42) 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Матрица ошибок:")
        print(confusion_matrix(y_test, y_pred))
        print("\nОтчет о классификации:")
        print(classification_report(y_test, y_pred))
        print("Точность:", round(accuracy_score(y_test, y_pred),2))