import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def print_section_header(title):
    print(f"\n{'='*10} {title} {'='*10}")

def load_and_prepare_data(file_path):
    print_section_header("1. Load and Prepare Data")
    board_feature_names = [f"{r}{c}" for r in range(8) for c in range(8)]
    all_headers = board_feature_names + ['first_score', 'second_score']

    try:
        data_df = pd.read_csv(file_path, header=None, names=all_headers)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{file_path}'.")
        exit()

    print(f"Kích thước dữ liệu gốc: {data_df.shape[0]} dòng, {data_df.shape[1]} cột.")
    if data_df.isnull().values.any():
        print("Cảnh báo: Phát hiện giá trị thiếu trong dữ liệu.")
        missing_counts = data_df.isnull().sum()
        print("Số giá trị thiếu theo cột (trước khi xử lý):")
        print(missing_counts[missing_counts > 0])

        for col in board_feature_names:
            if data_df[col].isnull().any():
                data_df[col] = data_df[col].fillna(0)
        
        if data_df[board_feature_names].isnull().values.any():
             print("Lỗi: Vẫn còn giá trị thiếu trong các cột đặc trưng bàn cờ sau khi fill. Kiểm tra lại dữ liệu.")
             exit()
        else:
            print("Giá trị thiếu trong các cột đặc trưng bàn cờ đã được fill bằng 0 (nếu có).")
        
        if data_df[['first_score', 'second_score']].isnull().values.any():
            print("Cảnh báo: Giá trị thiếu trong cột điểm. Các hàng này sẽ bị loại bỏ.")
            data_df.dropna(subset=['first_score', 'second_score'], inplace=True)
            print(f"Kích thước dữ liệu sau khi loại bỏ hàng thiếu điểm: {data_df.shape[0]} dòng.")
    else:
        print("Không phát hiện giá trị thiếu trong dữ liệu.")

    y_target = np.zeros(len(data_df), dtype=int)
    y_target[data_df['first_score'] > data_df['second_score']] = 1
    y_target[data_df['first_score'] < data_df['second_score']] = -1
    
    # CHỈ SỬ DỤNG 64 FEATURES BÀN CỜ CHO X_features
    X_features = data_df[board_feature_names].copy() 
    print(f"Sử dụng {X_features.shape[1]} features cho model (chỉ trạng thái bàn cờ).")
    return X_features, y_target

def split_data(X, y, test_size=0.33, random_state=42):
    print_section_header("2. Split Data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_model(model_name, model_class, X_train, y_train, **kwargs):
    print_section_header(f"3. Train {model_name} Model")
    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test):
    y_pred_labels = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    return y_pred_labels, y_pred_proba

def plot_confusion_matrix_custom(y_true, y_pred, class_labels, title_suffix):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {title_suffix}')
    plt.tight_layout()
    plt.show(block=False) # Thay đổi để không block nếu chạy script

def evaluate_model_performance(model_name, y_test_true, y_pred_labels, y_pred_proba, model_classes_):
    print_section_header(f"4. Evaluate {model_name} Model")

    accuracy = accuracy_score(y_test_true, y_pred_labels)
    print(f"Accuracy: {accuracy:.4f}")

    sorted_model_classes = sorted(list(model_classes_))
    target_names_report = [str(cls) for cls in sorted_model_classes]

    report = classification_report(y_test_true, y_pred_labels, labels=sorted_model_classes, target_names=target_names_report, zero_division=0)
    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix_custom(y_test_true, y_pred_labels, sorted_model_classes, model_name)

    if y_pred_proba is not None:
        try:
            unique_classes_in_test = np.unique(y_test_true)
            if len(unique_classes_in_test) > 1:
                y_true_for_roc, roc_labels = y_test_true, sorted_model_classes
                if model_name == "XGBoost":
                    label_mapping_xgb = {-1: 0, 0: 1, 1: 2}
                    y_true_for_roc = np.array([label_mapping_xgb.get(label, label) for label in y_test_true])
                    roc_labels = sorted(list(label_mapping_xgb.values()))
                
                if not all(label in roc_labels for label in np.unique(y_true_for_roc)):
                    pass 

                if len(np.unique(y_true_for_roc)) > 2:
                    roc_auc_ovr_macro = roc_auc_score(y_true_for_roc, y_pred_proba, multi_class='ovr', average='macro', labels=roc_labels)
                    print(f"ROC AUC Score (OvR, macro): {roc_auc_ovr_macro:.4f}")
                elif len(np.unique(y_true_for_roc)) == 2:
                    positive_class_val = 1
                    if model_name == "XGBoost": positive_class_val = label_mapping_xgb.get(1,1) # mapped value of 1
                    
                    if y_pred_proba.shape[1] == 2:
                        try:
                            pos_idx = roc_labels.index(positive_class_val)
                            roc_auc = roc_auc_score(y_true_for_roc, y_pred_proba[:, pos_idx])
                            print(f"ROC AUC Score (cho lớp {positive_class_val}): {roc_auc:.4f}")
                        except ValueError:
                             print(f"Cảnh báo: Không tìm thấy lớp dương {positive_class_val} trong roc_labels cho ROC AUC nhị phân.")
                    elif y_pred_proba.shape[1] > 2 and len(roc_labels) == y_pred_proba.shape[1]:
                        print(f"Cảnh báo: y_true_for_roc có 2 lớp nhưng y_pred_proba có {y_pred_proba.shape[1]} cột cho {model_name}.")
                        roc_auc_ovr_macro = roc_auc_score(y_true_for_roc, y_pred_proba, multi_class='ovr', average='macro', labels=roc_labels)
                        print(f"ROC AUC Score (OvR, macro, dù y_true_for_roc có 2 lớp): {roc_auc_ovr_macro:.4f}")

        except ValueError as e:
            print(f"Lỗi khi tính ROC AUC cho {model_name}: {e}")
    else:
        print(f"Không có y_pred_proba cho {model_name} để tính ROC AUC.")
    
def save_trained_model(model, file_path):
    print_section_header(f"5. Save Model to {file_path}")
    try:
        joblib.dump(model, file_path, compress=True)
        print(f"Mô hình đã được lưu vào '{file_path}'")
    except Exception as e:
        print(f"Lỗi khi lưu model: {e}")

def plot_model_feature_importances(model_name, model, feature_names_list):
    print_section_header(f"6. Feature Importances for {model_name}")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if len(importances) == 64 and len(feature_names_list) == 64: 
            importances_reshaped = importances.reshape([8, 8])
            plt.figure(figsize=(10, 8))
            sns.heatmap(importances_reshaped, annot=True, fmt=".3f", cmap="viridis",
                        xticklabels=[str(i) for i in range(8)],
                        yticklabels=[str(i) for i in range(8)])
            plt.xlabel("Cột (Board Y)")
            plt.ylabel("Hàng (Board X)")
            plt.title(f"Feature Importances ({model_name}) - Mức độ quan trọng của từng ô")
            plt.tight_layout()
            plt.show(block=False)

            sorted_indices = np.argsort(importances)[::-1]
            print("\nTop 10 features quan trọng nhất:")
            for i in range(min(10, len(feature_names_list))):
                print(f"{i+1}. Ô {feature_names_list[sorted_indices[i]]}: {importances[sorted_indices[i]]:.4f}")
        else:
            print(f"Số lượng feature importances ({len(importances)}) hoặc feature names ({len(feature_names_list)}) không khớp 64.")
    else:
        print(f"Model {model_name} không có thuộc tính 'feature_importances_'.")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_dir, 'data', 'othello_state_dataset.csv')
    
    X_features, y_target_orig = load_and_prepare_data(data_file_path)
    if X_features is None : return

    feature_names = list(X_features.columns)
    X_train, X_test, y_train_orig, y_test_orig = split_data(X_features, y_target_orig)

    new_dir = os.path.join(current_dir, '..', 'ai', 'ML_DL_models')

    rf_model = train_model(
        "Random Forest",
        RandomForestClassifier,
        X_train, y_train_orig,
        n_estimators=500, n_jobs=-1, random_state=42, class_weight='balanced'
    )
    y_pred_labels_rf, y_pred_proba_rf = make_predictions(rf_model, X_test)
    evaluate_model_performance("Random Forest", y_test_orig, y_pred_labels_rf, y_pred_proba_rf, rf_model.classes_)
    save_trained_model(rf_model, os.path.join(new_dir, 'rf_othello_classifier.pkl'))
    plot_model_feature_importances("Random Forest", rf_model, feature_names)

    label_mapping_xgb = {-1: 0, 0: 1, 1: 2}
    y_train_xgb_mapped = np.array([label_mapping_xgb.get(label) for label in y_train_orig])
    
    xgb_model_instance = train_model(
        "XGBoost",
        xgb.XGBClassifier,
        X_train, y_train_xgb_mapped,
        n_estimators=200, random_state=42,
        eval_metric='mlogloss',
        objective='multi:softprob'
    )
    y_pred_labels_xgb_mapped, y_pred_proba_xgb = make_predictions(xgb_model_instance, X_test)
    
    inverse_label_mapping_xgb = {v: k for k, v in label_mapping_xgb.items()}
    y_pred_labels_xgb_orig = np.array([inverse_label_mapping_xgb.get(label) for label in y_pred_labels_xgb_mapped])
    
    evaluate_model_performance("XGBoost", y_test_orig, y_pred_labels_xgb_orig, y_pred_proba_xgb, xgb_model_instance.classes_)
    save_trained_model(xgb_model_instance, os.path.join(new_dir, 'xgb_othello_classifier.pkl'))
    plot_model_feature_importances("XGBoost", xgb_model_instance, feature_names)

    print_section_header("All training and evaluations complete.")
    plt.show()

if __name__ == '__main__':
    main()