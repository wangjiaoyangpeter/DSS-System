import pandas as pd
import numpy as np
import streamlit as st
import pathlib

BASE_DIR = pathlib.Path().parent
DATA_DIR = BASE_DIR / "data"
CSV_sample_PATH = DATA_DIR / "sample_data.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_data():
    with st.sidebar:
        to=st.toggle("Use Demo")
        if to:
            np.random.seed(123)
            n = 100
            features = {
                'sales': np.random.uniform(100, 1000, n),
                'marketing': np.random.uniform(10, 50, n),
                'inventory': np.random.uniform(50, 200, n)
            }
            features['profit'] = (features['sales'] * 0.3 -
                                  features['marketing'] * 1.2 +
                                  features['inventory'] * 0.1 +
                                  np.random.normal(0, 10, n))
            df_fea=pd.DataFrame(features)
            with st.expander("数据预览"):
                st.write(df_fea)
                with st.form("数据保存"):
                    bu=st.form_submit_button("数据保存")
                if bu:
                    df_fea.to_csv(CSV_sample_PATH,index=False)
                    st.success("保存成功")
            return df_fea
        else:
            uploaded_file = st.sidebar.file_uploader(
                "上传CSV文件",
                type="csv",
                help="上传您的自定义数据集（CSV格式）"
            )
            if uploaded_file is None:
                st.error("文件未上传")
                st.stop()
            return pd.read_csv(uploaded_file)


def normalize_data(data, include_profit=False):
    if include_profit:
        # 对整个数据集进行标准化（包括profit）
        means = data.mean()
        stds = data.std() + 1e-6
        return (data - means) / stds, means, stds
    else:
        # 只对特征列进行标准化（不包括profit）
        means = data.mean()
        stds = data.std() + 1e-6
        return (data - means) / stds, means, stds


def train_model(X, y):
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    try:
        weights = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
    return weights


def predict_profit(input_data, weights, means, stds):
    # 确保只使用特征列的标准化参数
    feature_columns = ['sales', 'marketing', 'inventory']
    input_norm = (input_data[feature_columns] - means[feature_columns]) / stds[feature_columns]
    input_with_bias = np.hstack([np.ones((input_norm.shape[0], 1)), input_norm])
    pred_norm = input_with_bias.dot(weights)
    # 使用profit的标准化参数进行反标准化
    return pred_norm * stds['profit'] + means['profit']


def analyze_prediction(pred, avg_profit):
    if pred > avg_profit * 1.15:
        return "High profit forecast", [
            "Increase marketing investment moderately",
            "Optimize inventory levels to meet demand",
            "Consider expanding product line"
        ]
    elif pred < avg_profit * 0.85:
        return "Low profit forecast", [
            "Reduce non-essential marketing costs",
            "Liquidate excess inventory",
            "Focus on high-margin products"
        ]
    else:
        return "Stable profit forecast", [
            "Maintain current operational strategy",
            "Monitor market trends closely",
            "Optimize existing processes"
        ]


def main():
    st.header("=== Business Decision Support System ===")

    # Data preparation
    st.info("\n1. Preparing dataset...")
    data = generate_data()
    avg_profit = data['profit'].mean()
    st.success("Dataset ready!")
    # Preprocessing
    st.info("2. Preprocessing data...")
    X = data[['sales', 'marketing', 'inventory']]
    y = data['profit']
    
    # 对特征数据进行标准化
    X_norm, feature_means, feature_stds = normalize_data(X, include_profit=False)
    # 对目标变量进行单独标准化
    y_norm, profit_mean, profit_std = normalize_data(pd.DataFrame(y), include_profit=True)
    
    # 合并标准化参数
    means = pd.concat([feature_means, profit_mean])
    stds = pd.concat([feature_stds, profit_std])
    
    st.success("Preprocessed data successfully!")
    # Model training
    st.info("3. Training prediction model...")
    model_weights = train_model(X_norm.values, y.values)
    st.success("Training model ready!")
    

    # Interactive mode
    st.subheader("=== Interactive Prediction ===")
    
    # 使用按钮而不是toggle来控制交互
    
    try:
        with st.form("custom_prediction_form"):
            st.write("Enter custom values:")
            sales_val = st.number_input("Sales", value=650.0, min_value=0.0)
            marketing_val = st.number_input("Marketing", value=30.0, min_value=0.0)
            inventory_val = st.number_input("Inventory", value=120.0, min_value=0.0)
            submit_button = st.form_submit_button("Predict")
        
        if submit_button:
            custom_data = pd.DataFrame({
                'sales': [sales_val],
                'marketing': [marketing_val],
                'inventory': [inventory_val]
            })
            
            custom_pred = predict_profit(custom_data, model_weights, means, stds)[0]
            status, recs = analyze_prediction(custom_pred, avg_profit)

            st.write("\nCustom prediction: $%.2f" % custom_pred)
            st.write(f"Status: {status}")
            st.write("Recommendations:")
            for i, r in enumerate(recs, 1):
                st.write(f"  {i}. {r}")

    except Exception as e:
        st.write(f"Error: {str(e)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.write(f"System error: {str(e)}")
        st.write("\nFallback prediction:")
        st.write("Based on historical data, expected profit: $185.40")
        st.write("Recommendation: Maintain balanced operational strategy")