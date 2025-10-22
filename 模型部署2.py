import streamlit as st
import joblib
import numpy as np
import pandas as pd
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

def state_change_to_text(state_change):
    """将状态变化数值转换为文字描述"""
    mapping = {
        0: "No organ failure",
        1: "Organ failure remission", 
        2: "Persistent organ failure"
    }
    return mapping.get(state_change, "Unknown")

# Streamlit 界面
st.title("IPN Prediction Model with Probability Visualization")

# 创建三列布局
col1, col2, col3 = st.columns(3)

user_inputs = {}

with col1:
    st.subheader("Admission indicators")
    
    # 第一列：入院时的各项指标（去掉_admission尾缀显示）
    admission_indicators = [
        "Respiratory_system", "Renal_system", "Cardiovascular_system", 
        "Coagulation_system", "Hepatic_system", "Neurological_system",
        "IAP", "Ca2+"
    ]
    
    for indicator in admission_indicators:
        feature_name = f"{indicator}_admission" if indicator not in ["IAP", "Ca2+"] else indicator
        properties = original_feature_ranges[feature_name]
        
        # 显示名称去掉尾缀
        display_name = indicator
        
        if indicator in ["Respiratory_system", "Renal_system", "Cardiovascular_system", 
                        "Coagulation_system", "Hepatic_system", "Neurological_system"]:
            # SOFA系统评分
            user_inputs[feature_name] = st.number_input(
                label=f"{display_name} (0-4)",
                min_value=0,
                max_value=4,
                value=int(properties["default"]),
                key=feature_name
            )
        else:
            # 其他指标
            user_inputs[feature_name] = st.number_input(
                label=f"{display_name} ({properties['min']} - {properties['max']})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                key=feature_name
            )

with col2:
    st.subheader("Week 2 indicators")
    
    # 第二列：病程第二周时的各项指标（去掉_t2尾缀显示）
    week2_indicators = [
        "Respiratory_system", "Renal_system", "Cardiovascular_system",
        "IAP", "EN_energy_intake", "Lymphocyte", "CRP"
    ]
    
    for indicator in week2_indicators:
        feature_name = f"{indicator}_t2"
        properties = original_feature_ranges[feature_name]
        
        # 显示名称去掉尾缀
        display_name = indicator
        
        if indicator in ["Respiratory_system", "Renal_system", "Cardiovascular_system"]:
            # SOFA系统评分
            user_inputs[feature_name] = st.number_input(
                label=f"{display_name} (0-4)",
                min_value=0,
                max_value=4,
                value=int(properties["default"]),
                key=feature_name
            )
        else:
            # 其他指标
            user_inputs[feature_name] = st.number_input(
                label=f"{display_name} ({properties['min']} - {properties['max']})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                key=feature_name
            )

with col3:
    st.subheader("Imaging indicator")
    
    # 第三列：影像学指标
    imaging_indicators = ["CTSI"]
    
    for indicator in imaging_indicators:
        properties = original_feature_ranges[indicator]
        user_inputs[indicator] = st.number_input(
            label=f"{indicator} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            key=indicator
        )

# 预测与概率可视化
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
    
    # 显示计算出的衍生特征 - 状态变化用文字显示
    st.subheader("Calculated Derived Features")
    st.write(f"SOFA Admission Total: {derived_features['SOFA_admission']}")
    st.write(f"IAP Dynamic Severity: {derived_features['IAP_dynamic_severity']:.2f}")
    
    # 将状态变化数值转换为文字描述
    respiratory_state_text = state_change_to_text(derived_features['Respiratory_system_state_change'])
    renal_state_text = state_change_to_text(derived_features['Renal_system_state_change'])
    cardiovascular_state_text = state_change_to_text(derived_features['Cardiovascular_system_state_change'])
    
    st.write(f"Respiratory System State Change: {respiratory_state_text}")
    st.write(f"Renal System State Change: {renal_state_text}")
    st.write(f"Cardiovascular System State Change: {cardiovascular_state_text}")
    
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
    st.pyplot(fig)
    
    # 生成基于预测结果的建议
    if predicted_class == 1:  # 如果预测为IPN
        advice = (
            f"According to our model, your risk of IPN is high. "
            f"The probability of you having IPN is {probability:.1f}%. "
        )
    else:  # 如果预测为无IPN
        advice = (
            f"According to our model, your risk of IPN is low. "
            f"The probability of you not having IPN is {probability:.1f}%. "
        )
    
    st.write(advice)
    
    # 可视化预测概率 - 使用方案二的绘图方法
    sample_prob = {
        'No IPN': predicted_proba[0],  # 类别0的概率
        'IPN': predicted_proba[1]  # 类别1的概率
    }

    # 设置图形大小
    plt.figure(figsize=(10, 3))

    # 创建条形图
    bars = plt.barh(['No IPN', 'IPN'], 
                    [sample_prob['No IPN'], sample_prob['IPN']], 
                    color=['#512b58', '#fe346e'])

    # 添加标题和标签，设置字体加粗和字体大小
    plt.title("Prediction Probability for Patient", fontsize=20, fontweight='bold')
    plt.xlabel("Probability", fontsize=14, fontweight='bold')
    plt.ylabel("Outcome", fontsize=14, fontweight='bold')

    # 添加概率文本标签，调整位置避免重叠，设置字体加粗
    for i, v in enumerate([sample_prob['No IPN'], sample_prob['IPN']]):
        plt.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')

    # 隐藏上、右、下边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    # 显示图形
    st.pyplot(plt)