# 预测与概率可视化
if st.button("Predict"):
    try:
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
        text = f"Based on feature values, predicted probability of IPN is {ipn_probability:.2f}%"
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
        
        # 生成基于预测结果的建议 - 基于IPN概率
        if ipn_probability >= 50:  # 如果IPN概率大于等于50%
            advice = (
                f"According to our model, your risk of IPN is high. "
                f"The probability of you having IPN is {ipn_probability:.1f}%. "
            )
        else:  # 如果IPN概率小于50%
            advice = (
                f"According to our model, your risk of IPN is low. "
                f"The probability of you having IPN is {ipn_probability:.1f}%. "
            )
        
        st.write(advice)
        
        # 可视化预测概率
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
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.info("Please check if all input values are within the specified ranges and try again.")