import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('rf.pkl')

# 定义原始特征的输入范围
original_feature_ranges = {
    # IAP相关
    "IAP_admission": {"type": "numerical", "min": 6, "max": 35, "default": 18},
    "IAP_t2": {"type": "numerical", "min": 6, "max": 35, "default": 15},
    
    # 实验室指标
    "Ca2+_admission": {"type": "numerical", "min": 1.6, "max": 2.7, "default": 1.95},
    "EN_energy_intake_t2": {"type": "numerical", "min": 0, "max": 2000, "default": 1000},
    "Lymphocyte_t2": {"type": "numerical", "min": 0.1, "max": 3, "default": 0.8},
    "CRP_t2": {"type": "numerical", "min": 0.5, "max": 385, "default": 150},
    "CTSI": {"type": "numerical", "min": 4, "max": 10, "default": 8},
    
    # SOFA系统评分 - 入院时（所有六个系统）
    "Respiratory_system_admission": {"type": "numerical", "min": 0, "max": 4, "default": 3},
    "Renal_system_admission": {"type": "numerical", "min": 0, "max": 4, "default": 2},
    "Cardiovascular_system_admission": {"type": "numerical", "min": 0, "max": 4, "default": 3},
    "Coagulation_system_admission": {"type": "numerical", "min": 0, "max": 4, "default": 1},
    "Hepatic_system_admission": {"type": "numerical", "min": 0, "max": 4, "default": 2},
    "Neurological_system_admission": {"type": "numerical", "min": 0, "max": 4, "default": 0},
    
    # SOFA系统评分 - 第二周（只需要前三个系统）
    "Respiratory_system_t2": {"type": "numerical", "min": 0, "max": 4, "default": 3},
    "Renal_system_t2": {"type": "numerical", "min": 0, "max": 4, "default": 0},
    "Cardiovascular_system_t2": {"type": "numerical", "min": 0, "max": 4, "default": 3},
}

# 模型所需的特征顺序（确保与训练时一致）
model_features = [
    "SOFA_admission", "IAP_admission", "Ca2+_admission", "EN_energy_intake_t2", 
    "Lymphocyte_t2", "CRP_t2", "CTSI", "Respiratory_system_state_change", "Renal_system_state_change", 
    "Cardiovascular_system_state_change", "IAP_dynamic_severity"
]

def calculate_derived_features(input_dict):
    """计算衍生特征"""
    derived = {}
    
    # 1. 计算SOFA总分（入院时）- 使用所有六个系统
    sofa_systems = ['Respiratory', 'Renal', 'Cardiovascular', 'Coagulation', 'Hepatic', 'Neurological']
    derived['SOFA_admission'] = sum(input_dict[f'{system}_system_admission'] for system in sofa_systems)
    
    # 2. 计算IAP动态严重度
    derived['IAP_dynamic_severity'] = input_dict['IAP_admission'] * (input_dict['IAP_t2'] - input_dict['IAP_admission'])
    
    # 3. 计算三个系统的状态变化（只需要呼吸、肾脏和心血管系统）
    systems_for_state_change = ['Respiratory', 'Renal', 'Cardiovascular']
    
    for system in systems_for_state_change:
        admission = input_dict[f'{system}_system_admission']
        t2 = input_dict[f'{system}_system_t2']
        
        if admission < 2 and t2 < 2:
            state_change = 0  # 无衰竭
        elif admission >= 2 and t2 < 2:
            state_change = 1  # 缓解衰竭
        elif admission < 2 and t2 >= 2:
            state_change = 2  # 新发衰竭
        else:  # admission >= 2 and t2 >= 2
            state_change = 2  # 持续性衰竭
            
        derived[f'{system}_system_state_change'] = state_change
    
    return derived

# Streamlit 界面
st.title("IPN Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
user_inputs = {}

# 创建三列布局
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("IAP and Laboratory Values")
    for feature in ["IAP_admission", "IAP_t2", "Ca2+_admission", "EN_energy_intake_t2", 
                   "Lymphocyte_t2", "CRP_t2", "CTSI"]:
        properties = original_feature_ranges[feature]
        user_inputs[feature] = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            key=feature
        )

with col2:
    st.subheader("SOFA Scores - Admission")
    for feature in ["Respiratory_system_admission", "Renal_system_admission", 
                   "Cardiovascular_system_admission", "Coagulation_system_admission",
                   "Hepatic_system_admission", "Neurological_system_admission"]:
        properties = original_feature_ranges[feature]
        user_inputs[feature] = st.number_input(
            label=f"{feature} (0-4)",
            min_value=0,
            max_value=4,
            value=int(properties["default"]),
            key=feature
        )

with col3:
    st.subheader("SOFA Scores - Week 2")
    # 只需要三个系统的第二周评分
    for feature in ["Respiratory_system_t2", "Renal_system_t2", "Cardiovascular_system_t2"]:
        properties = original_feature_ranges[feature]
        user_inputs[feature] = st.number_input(
            label=f"{feature} (0-4)",
            min_value=0,
            max_value=4,
            value=int(properties["default"]),
            key=feature
        )

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 计算衍生特征
    derived_features = calculate_derived_features(user_inputs)
    
    # 准备模型输入
    model_input = []
    for feature in model_features:
        if feature in user_inputs:
            model_input.append(user_inputs[feature])
        else:
            model_input.append(derived_features[feature])
    
    features = np.array([model_input])
    
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    
    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100
    
    # 显示计算出的衍生特征
    st.subheader("Calculated Derived Features")
    st.write(f"SOFA Admission Total: {derived_features['SOFA_admission']}")
    st.write(f"IAP Dynamic Severity: {derived_features['IAP_dynamic_severity']:.2f}")
    st.write(f"Respiratory System State Change: {derived_features['Respiratory_system_state_change']}")
    st.write(f"Renal System State Change: {derived_features['Renal_system_state_change']}")
    st.write(f"Cardiovascular System State Change: {derived_features['Cardiovascular_system_state_change']}")
    
    # 显示预测结果
    st.subheader("Prediction Result")
    text = f"Based on feature values, predicted possibility of IPN is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")
    
    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([model_input], columns=model_features))
    
    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([model_input], columns=model_features),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
    
    # 显示特征重要性（可选）
    st.subheader("Feature Importance (SHAP Values)")
    shap_summary = shap.summary_plot(shap_values[:,:,class_index], 
                                   pd.DataFrame([model_input], columns=model_features),
                                   plot_type="bar", show=False)
    plt.savefig("shap_summary.png", bbox_inches='tight', dpi=300)
    st.image("shap_summary.png")