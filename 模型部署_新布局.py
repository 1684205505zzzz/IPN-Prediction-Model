import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置页面为宽屏模式
st.set_page_config(
    page_title="IPN Prediction Model",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main > div {
        max-width: 100%;
        padding-left: 3%;
        padding-right: 3%;
    }
    
    .stColumns {
        gap: 2rem;
    }
    
    .stNumberInput {
        min-width: 120px;
    }
    
    h1 {
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    
    .stHeader {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 加载保存的LightGBM模型
model = joblib.load('lightgbm.pkl')

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
    state_mapping = {
        0: "No organ failure",
        1: "Organ failure remission", 
        2: "Persistent organ failure"
    }
    return state_mapping.get(state_change, "Unknown")

# Streamlit 界面
st.title("IPN Prediction Model with Probability Visualization")

# 创建左右两栏布局 - 使用更宽的比例
left_col, right_col = st.columns([1.2, 0.8])

# 左侧栏：输入数据
with left_col:
    st.header("Enter the following feature values:")
    user_inputs = {}

    # 创建三列布局用于输入
    col1, col2, col3 = st.columns([1, 1, 0.8])

    # 第一列：Admission indicators
    with col1:
        st.subheader("Admission indicators")
        
        # SOFA系统评分 - 入院时
        admission_sofa_features = [
            "Respiratory_system", "Renal_system", "Cardiovascular_system", 
            "Coagulation_system", "Hepatic_system", "Neurological_system"
        ]
        
        for feature_base in admission_sofa_features:
            feature_name = f"{feature_base}_admission"
            properties = original_feature_ranges[feature_name]
            user_inputs[feature_name] = st.number_input(
                label=f"{feature_base} (0-4)",
                min_value=0,
                max_value=4,
                value=int(properties["default"]),
                key=feature_name
            )
        
        # IAP和Ca2+入院指标
        iap_properties = original_feature_ranges["IAP_admission"]
        user_inputs["IAP_admission"] = st.number_input(
            label=f"IAP ({iap_properties['min']} - {iap_properties['max']})",
            min_value=float(iap_properties["min"]),
            max_value=float(iap_properties["max"]),
            value=float(iap_properties["default"]),
            key="IAP_admission"
        )
        
        ca_properties = original_feature_ranges["Ca2+_admission"]
        user_inputs["Ca2+_admission"] = st.number_input(
            label=f"Ca2+ ({ca_properties['min']} - {ca_properties['max']})",
            min_value=float(ca_properties["min"]),
            max_value=float(ca_properties["max"]),
            value=float(ca_properties["default"]),
            key="Ca2+_admission"
        )

    # 第二列：Week 2 indicators
    with col2:
        st.subheader("Week 2 indicators")
        
        # SOFA系统评分 - 第二周
        week2_sofa_features = ["Respiratory_system", "Renal_system", "Cardiovascular_system"]
        
        for feature_base in week2_sofa_features:
            feature_name = f"{feature_base}_t2"
            properties = original_feature_ranges[feature_name]
            user_inputs[feature_name] = st.number_input(
                label=f"{feature_base} (0-4)",
                min_value=0,
                max_value=4,
                value=int(properties["default"]),
                key=feature_name
            )
        
        # IAP第二周
        iap_t2_properties = original_feature_ranges["IAP_t2"]
        user_inputs["IAP_t2"] = st.number_input(
            label=f"IAP ({iap_t2_properties['min']} - {iap_t2_properties['max']})",
            min_value=float(iap_t2_properties["min"]),
            max_value=float(iap_t2_properties["max"]),
            value=float(iap_t2_properties["default"]),
            key="IAP_t2"
        )
        
        # 其他第二周指标
        en_properties = original_feature_ranges["EN_energy_intake_t2"]
        user_inputs["EN_energy_intake_t2"] = st.number_input(
            label=f"EN energy intake ({en_properties['min']} - {en_properties['max']})",
            min_value=float(en_properties["min"]),
            max_value=float(en_properties["max"]),
            value=float(en_properties["default"]),
            key="EN_energy_intake_t2"
        )
        
        lymph_properties = original_feature_ranges["Lymphocyte_t2"]
        user_inputs["Lymphocyte_t2"] = st.number_input(
            label=f"Lymphocyte ({lymph_properties['min']} - {lymph_properties['max']})",
            min_value=float(lymph_properties["min"]),
            max_value=float(lymph_properties["max"]),
            value=float(lymph_properties["default"]),
            key="Lymphocyte_t2"
        )
        
        crp_properties = original_feature_ranges["CRP_t2"]
        user_inputs["CRP_t2"] = st.number_input(
            label=f"CRP ({crp_properties['min']} - {crp_properties['max']})",
            min_value=float(crp_properties["min"]),
            max_value=float(crp_properties["max"]),
            value=float(crp_properties["default"]),
            key="CRP_t2"
        )

    # 第三列：Imaging indicator
    with col3:
        st.subheader("Imaging indicator")
        
        ctsi_properties = original_feature_ranges["CTSI"]
        user_inputs["CTSI"] = st.number_input(
            label=f"CTSI ({ctsi_properties['min']} - {ctsi_properties['max']})",
            min_value=float(ctsi_properties["min"]),
            max_value=float(ctsi_properties["max"]),
            value=float(ctsi_properties["default"]),
            key="CTSI"
        )
    
    # 预测按钮放在左侧栏底部
    if st.button("Predict", key="predict_button", use_container_width=True):
        # 将用户输入存储到session state中，以便右侧栏可以访问
        st.session_state.user_inputs = user_inputs
        st.session_state.predict_clicked = True

# 右侧栏：显示结果
with right_col:
    st.header("Prediction Results")
    
    # 检查是否点击了预测按钮
    if hasattr(st.session_state, 'predict_clicked') and st.session_state.predict_clicked:
        try:
            user_inputs = st.session_state.user_inputs
            
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
            
            # 提取IPN的概率（类别1的概率）
            ipn_probability = predicted_proba[1] * 100
            
            # 显示计算出的衍生特征
            st.subheader("Calculated Derived Features")
            st.write(f"SOFA Admission Total: {derived_features['SOFA_admission']}")
            st.write(f"IAP Dynamic Severity: {derived_features['IAP_dynamic_severity']:.2f}")
            st.write(f"Respiratory System State Change: {state_change_to_text(derived_features['Respiratory_system_state_change'])}")
            st.write(f"Renal System State Change: {state_change_to_text(derived_features['Renal_system_state_change'])}")
            st.write(f"Cardiovascular System State Change: {state_change_to_text(derived_features['Cardiovascular_system_state_change'])}")
            
            # 显示预测结果 - 统一显示IPN的概率
            st.subheader("Prediction Result")
            
            # 使用彩色标记显示结果
            if ipn_probability >= 50:
                st.error(f"⚠️ High Risk of IPN: {ipn_probability:.2f}%")
            else:
                st.success(f"✅ Low Risk of IPN: {ipn_probability:.2f}%")
            
            # 生成基于预测结果的建议 - 基于IPN概率
            st.subheader("Clinical Recommendation")
            if ipn_probability >= 50:
                advice = (
                    f"According to our model, the patient has a high risk of IPN. "
                    f"The probability of IPN is {ipn_probability:.1f}%. "
                    "Consider close monitoring and appropriate interventions."
                )
                st.warning(advice)
            else:
                advice = (
                    f"According to our model, the patient has a low risk of IPN. "
                    f"The probability of IPN is {ipn_probability:.1f}%. "
                    "Continue with standard monitoring protocols."
                )
                st.info(advice)
            
            # 可视化预测概率
            st.subheader("Probability Visualization")
            
            sample_prob = {
                'No IPN': predicted_proba[0],  # 类别0的概率
                'IPN': predicted_proba[1]  # 类别1的概率
            }

            # 设置图形大小
            fig, ax = plt.subplots(figsize=(10, 3))

            # 创建条形图
            bars = ax.barh(['No IPN', 'IPN'], 
                            [sample_prob['No IPN'], sample_prob['IPN']], 
                            color=['#512b58', '#fe346e'])

            # 添加标题和标签，设置字体加粗和字体大小
            ax.set_title("Prediction Probability for Patient", fontsize=20, fontweight='bold')
            ax.set_xlabel("Probability", fontsize=14, fontweight='bold')
            ax.set_ylabel("Outcome", fontsize=14, fontweight='bold')

            # 添加概率文本标签，调整位置避免重叠，设置字体加粗
            for i, v in enumerate([sample_prob['No IPN'], sample_prob['IPN']]):
                ax.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')

            # 隐藏上、右、下边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # 显示图形
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please check if all input values are within the specified ranges and try again.")
    else:
        # 当还没有点击预测按钮时显示提示信息
        st.info("👈 Please enter the patient data in the left panel and click 'Predict' to see the results here.")


