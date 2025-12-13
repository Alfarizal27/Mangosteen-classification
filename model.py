import cv2
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import joblib

# path configuration

#base_path = "C:\kuliah\Semester 5\Machine learning\project 3 matkul\Mangosteen.v1i.multiclass"
#
##train_path = os.path.join(base_path, "train")
##image_train = train_path
##label_train = os.path.join(train_path, "train_classes.csv")
##
##test_path = os.path.join(base_path, "test")
##image_test = test_path
##label_test = os.path.join(test_path, "test_classes.csv")
#
#manggis_path = os.path.join(base_path, "all_dataset")
#label_path = os.path.join(manggis_path, "manggis_label.csv")
#image_path = manggis_path

# load data function

# load csv + one hot -> label tunggal
def load_label(label_path):
    df = pd.read_csv(label_path, header=None, skiprows=1)
    df.columns = ["filename", "l1", "l2", "l3", "l4", "l5"]
    
    df["label"] = df[["l1", "l2", "l3", "l4", "l5"]].idxmax(axis=1)
    df["label"] = df["label"].str.replace("l", "").astype(int)
    
    return df

# load gambar dari all_dataset
def load_images(folder, df):
    X = []
    y = []
        
    for _, row in df.iterrows():
        filename = row["filename"]
        label = row["label"]
        
        
        img_path = os.path.join(folder, filename)
        if not os.path.exists(img_path):
            print("file tidak ditemukan", img_path)
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print("gambar gagal dibaca", img_path)
            continue
        
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.flatten()
        
        X.append(img)
        y.append(label)
        
    return np.array(X), np.array(y)

# Train dan validate dengan K-Fold
def train_kfold(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    model = DecisionTreeClassifier()

    best_model = None
    best_acc = 0
    
    fold = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        # evaluate
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        
        print("Accuracy:", acc)
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            
        fold += 1
    
    joblib.dump(best_model, "decision_tree_manggis.pkl")
    
    return best_model

def main():
    
    folder = "C:\kuliah\Semester 5\Machine learning\project 3 matkul\Mangosteen.v1i.multiclass/all_dataset"
    csv_path = os.path.join(folder, "manggis_label.csv")
    
    df = load_label(csv_path)
    X, y = load_images(folder, df)
    
    print("total dataset:", len(X))
    train_kfold(X, y)
    
if __name__ == "__main__":
    main()
    