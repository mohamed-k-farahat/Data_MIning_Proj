import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
import shap
from sklearn.ensemble import RandomForestClassifier  # Example model

def show_churn():
    st.title("Customer Churn Prediction Analysis")
    
    # Create tabs for different sections with improved styling that works across environments
    st.markdown(
        """
        <style>
        /* Fix for tab styling to ensure they're flushed with the page */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px;
            margin-left: -1rem;
            margin-right: -1rem;
            padding-left: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            margin: 0px !important;
            padding: 1rem 1rem;
            background-color: transparent;
            color: white !important; /* Force white text for visibility in dark mode */
            font-weight: 500;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(255, 255, 255, 0.1); /* Lighter hover in dark mode */
        }
        
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: rgba(255, 255, 255, 0.2); /* Visible highlight in dark mode */
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: rgba(255, 255, 255, 0.15);
        }
        
        /* Ensure all text is visible in dark mode */
        .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, li, span {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    tabs = st.tabs(["Prediction Tool", "Model Insights", "What-If Analysis", "Retention Strategies"])
    
    # Load the model
    try:
        model = joblib.load("models/churn_model.joblib")
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Using a placeholder model for demonstration purposes.")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model_loaded = False
    
    # Tab 1: Prediction Tool
    with tabs[0]:
        st.header("Predict Customer Churn Risk")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Information")
            age = st.slider("Age", min_value=18, max_value=100, value=35)
            tenure = st.slider("Tenure (Years with Company)", min_value=0, max_value=50, value=5)
            sex = st.radio("Gender", options=["Male", "Female"])
            sex_mapped = 1 if sex == "Male" else 0
            
            # Additional features for enhanced prediction
            with st.expander("Advanced Features (Optional)"):
                monthly_charges = st.slider("Monthly Charges ($)", min_value=0, max_value=200, value=70)
                total_charges = st.slider("Total Charges to Date ($)", min_value=0, max_value=8000, value=monthly_charges * tenure * 12)
                contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                
                # Map contract type to numeric
                if contract_type == "Month-to-month":
                    contract_mapped = 0
                elif contract_type == "One year":
                    contract_mapped = 1
                else:
                    contract_mapped = 2
        
        with col2:
            st.subheader("Service Information")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
            # Map internet service to numeric
            if internet_service == "DSL":
                internet_mapped = 1
            elif internet_service == "Fiber optic":
                internet_mapped = 2
            else:
                internet_mapped = 0
                
            payment_method = st.selectbox("Payment Method", 
                                        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            
            # Map payment method to numeric
            if payment_method == "Electronic check":
                payment_mapped = 0
            elif payment_method == "Mailed check":
                payment_mapped = 1
            elif payment_method == "Bank transfer":
                payment_mapped = 2
            else:
                payment_mapped = 3
                
            has_phone = st.checkbox("Has Phone Service", value=True)
            has_online_security = st.checkbox("Has Online Security", value=False)
            has_tech_support = st.checkbox("Has Tech Support", value=False)
        
        # Use basic features for prediction if model is loaded
        # For demo, we'll use a simple input structure, but in reality, adapt to your model's expected input
        if model_loaded:
            input_data = np.array([[age, tenure, sex_mapped]])
        else:
            # For the placeholder model, create a more comprehensive input
            # You should adjust these features to match your actual model
            input_data = np.array([[
                age, tenure, sex_mapped, monthly_charges, total_charges, 
                contract_mapped, internet_mapped, payment_mapped,
                int(has_phone), int(has_online_security), int(has_tech_support)
            ]])
            
            # Train the placeholder model with dummy data
            X_dummy = np.random.rand(1000, input_data.shape[1])
            y_dummy = np.random.choice([0, 1], size=1000)
            model.fit(X_dummy, y_dummy)
        
        st.markdown("---")
        
        # Prediction button
        if st.button("Predict Churn Probability", key="predict_btn"):
            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0][1]
                
                # Display prediction with gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = proba * 100,
                    title = {'text': "Churn Risk"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                st.plotly_chart(fig)
                
                # Prediction text
                if prediction == 1:
                    st.error(f"⚠️ **High Risk of Churn** (Probability: {proba:.2%})")
                    st.markdown("### Recommended Actions:")
                    st.markdown("""
                    - Immediate outreach with retention offers
                    - Conduct satisfaction survey
                    - Offer service upgrade or discount
                    - Review billing history for possible issues
                    """)
                else:
                    st.success(f"✅ **Low Risk of Churn** (Retention Probability: {1-proba:.2%})")
                    st.markdown("### Recommended Actions:")
                    st.markdown("""
                    - Regular check-ins to maintain satisfaction
                    - Consider cross-selling opportunities
                    - Enroll in loyalty program if not already
                    - Early renewal incentives for contracts
                    """)
            except Exception as e:
                st.error(f"Error in prediction: {e}")
        
    # Tab 2: Model Insights
    with tabs[1]:
        st.header("Model Performance & Feature Importance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Accuracy Metrics")
            
            # Create sample metrics for demonstration
            metrics = {
                "Accuracy": 0.85,
                "Precision": 0.78,
                "Recall": 0.83,
                "F1 Score": 0.80,
                "AUC-ROC": 0.89
            }
            
            # Create metrics visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=['#1E88E5', '#42A5F5', '#64B5F6', '#90CAF9', '#BBDEFB'],
                text=[f"{v:.2%}" for v in metrics.values()],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Model Performance Metrics",
                xaxis_title="Metric",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                template="plotly_white"
            )
            
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Feature Importance")
            
            # Create sample feature importance for demonstration
            if model_loaded:
                try:
                    feature_importance = model.feature_importances_
                    feature_names = ["Age", "Tenure", "Gender"]
                except:
                    feature_importance = np.array([0.3, 0.5, 0.2])
                    feature_names = ["Age", "Tenure", "Gender"]
            else:
                feature_importance = model.feature_importances_
                feature_names = [
                    "Age", "Tenure", "Gender", "Monthly Charges", "Total Charges", 
                    "Contract Type", "Internet Service", "Payment Method",
                    "Phone Service", "Online Security", "Tech Support"
                ]
                
            # Sort feature importance
            sorted_idx = np.argsort(feature_importance)
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_importance = feature_importance[sorted_idx]
            
            # Create feature importance bar chart
            fig = px.bar(
                x=sorted_importance,
                y=sorted_features,
                orientation='h',
                title="Feature Importance",
                labels={"x": "Importance", "y": "Feature"},
                color=sorted_importance,
                color_continuous_scale="Blues"
            )
            
            st.plotly_chart(fig)
            
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        
        # Create sample confusion matrix
        cm = np.array([[85, 15], [20, 80]])
        
        fig, ax = plt.figure(figsize=(6, 5)), plt.axes()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        ax.xaxis.set_ticklabels(['Retained', 'Churned'])
        ax.yaxis.set_ticklabels(['Retained', 'Churned'])
        st.pyplot(fig)
        
        # ROC Curve
        st.subheader("ROC Curve")
        
        # Create sample ROC curve data
        fpr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        tpr = np.array([0, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 1.0])
        roc_auc = auc(fpr, tpr)
        
        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC = {roc_auc:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        fig.update_layout(
            template='plotly_white',
            xaxis=dict(range=[0, 1], constrain='domain'),
            yaxis=dict(range=[0, 1], constrain='domain')
        )
        
        st.plotly_chart(fig)
            
    # Tab 3: What-If Analysis
    with tabs[2]:
        st.header("What-If Analysis")
        
        st.write("""
        Explore how different customer characteristics impact churn probability.
        Adjust the parameters below to see how they affect the likelihood of customer churn.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Baseline Customer")
            baseline_age = 35
            baseline_tenure = 1
            baseline_gender = "Male"
            
            st.info(f"""
            **Baseline Profile:**
            - Age: {baseline_age}
            - Tenure: {baseline_tenure} years
            - Gender: {baseline_gender}
            - Contract: Month-to-month
            - Internet: Fiber optic
            - Payment: Electronic check
            """)
            
            # Create baseline input
            baseline_input = np.array([[baseline_age, baseline_tenure, 1]])
            
            # Get baseline prediction
            if model_loaded:
                baseline_churn_prob = model.predict_proba(baseline_input)[0][1]
            else:
                baseline_churn_prob = 0.7  # High risk for short tenure
        
        with col2:
            st.subheader("Adjust Parameters")
            new_tenure = st.slider("New Tenure (Years)", min_value=0, max_value=10, value=5, step=1)
            contract_change = st.selectbox("Change Contract To", ["Month-to-month", "One year", "Two year"])
            
            # Create adjusted input
            adjusted_input = np.array([[baseline_age, new_tenure, 1]])
            
            # Calculate new probability
            if model_loaded:
                new_churn_prob = model.predict_proba(adjusted_input)[0][1]
            else:
                # Simplified logic for demonstration
                tenure_factor = min(0.9, new_tenure * 0.1)  # Higher tenure = lower churn
                contract_factor = 0.0
                if contract_change == "One year":
                    contract_factor = 0.2
                elif contract_change == "Two year":
                    contract_factor = 0.4
                    
                new_churn_prob = max(0.1, baseline_churn_prob - tenure_factor - contract_factor)
            
            # Display probability change
            delta = baseline_churn_prob - new_churn_prob
            delta_percent = (delta / baseline_churn_prob) * 100
            
            st.metric(
                label="Churn Probability",
                value=f"{new_churn_prob:.2%}",
                delta=f"{-delta_percent:.1f}%" if delta > 0 else f"{abs(delta_percent):.1f}%",
                delta_color="inverse"
            )
            
        # Tenure impact visualization
        st.subheader("Impact of Tenure on Churn Probability")
        
        # Generate data for tenure impact
        tenures = list(range(0, 11))
        churn_probs = []
        
        for t in tenures:
            input_data = np.array([[baseline_age, t, 1]])
            if model_loaded:
                prob = model.predict_proba(input_data)[0][1]
            else:
                # Simple formula for demo purposes
                prob = max(0.1, 0.7 - (t * 0.06))
            churn_probs.append(prob)
            
        # Create line chart
        fig = px.line(
            x=tenures, y=churn_probs,
            markers=True,
            labels={"x": "Tenure (Years)", "y": "Churn Probability"},
            title="How Tenure Affects Churn Probability"
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            yaxis=dict(tickformat='.0%'),
            hovermode="x unified"
        )
        
        # Add area to highlight current selection
        fig.add_vrect(
            x0=new_tenure-0.2, x1=new_tenure+0.2,
            fillcolor="green", opacity=0.25,
            layer="below", line_width=0
        )
        
        st.plotly_chart(fig)
        
        # Contract impact visualization
        st.subheader("Impact of Contract Type on Churn Probability")
        
        contract_types = ["Month-to-month", "One year", "Two year"]
        contract_probs = [0.6, 0.35, 0.15]  # Sample probabilities
        
        fig = px.bar(
            x=contract_types, y=contract_probs,
            labels={"x": "Contract Type", "y": "Churn Probability"},
            title="How Contract Type Affects Churn Probability",
            color=contract_probs,
            color_continuous_scale="Blues_r"
        )
        
        fig.update_layout(
            yaxis=dict(tickformat='.0%')
        )
        
        # Highlight selected contract
        selected_idx = contract_types.index(contract_change)
        colors = ['lightblue'] * len(contract_types)
        colors[selected_idx] = 'green'
        fig.data[0].marker.color = colors
        
        st.plotly_chart(fig)
            
    # Tab 4: Retention Strategies
    with tabs[3]:
        st.header("Customer Retention Strategies")
        
        st.write("""
        Based on our churn analysis, we've identified effective retention strategies 
        targeted at different customer segments.
        """)
        
        # Segment-based strategies
        st.subheader("Segment-Based Retention Strategies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### High-Risk New Customers
            
            **Profile:**
            - Less than 1 year tenure
            - Month-to-month contracts
            - Higher monthly charges
            
            **Strategies:**
            - Welcome calls and onboarding support
            - First 3-month satisfaction guarantee
            - Incentives for annual contract conversion
            - Personalized usage recommendations
            """)
            
            st.markdown("""
            ### Price-Sensitive Customers
            
            **Profile:**
            - Medium-high churn risk
            - Responsive to price changes
            - Often compare with competitors
            
            **Strategies:**
            - Competitive price matching
            - Bundled services at discount
            - Transparent billing practices
            - Value-added services at no extra cost
            """)
        
        with col2:
            st.markdown("""
            ### Long-Term at Risk
            
            **Profile:**
            - 3+ years tenure
            - Recently increased service usage
            - Contacted support in last 3 months
            
            **Strategies:**
            - Loyalty rewards program
            - Priority customer service
            - Free service upgrades
            - Personal account manager
            """)
            
            st.markdown("""
            ### Service Issue Customers
            
            **Profile:**
            - Multiple support tickets
            - Service interruptions
            - Billing disputes
            
            **Strategies:**
            - Proactive service quality monitoring
            - Service restoration prioritization
            - Make-good compensation for issues
            - Regular check-ins after resolutions
            """)
        
        # ROI of retention efforts
        st.subheader("ROI of Retention Efforts")
        
        # Sample data for ROI visualization
        strategies = ["Loyalty Program", "Service Upgrade", "Price Discount", "Dedicated Support"]
        costs = [50, 75, 100, 150]  # Cost per customer
        retention_rates = [0.70, 0.82, 0.85, 0.90]  # Retention rate
        customer_values = [500, 500, 500, 500]  # Customer lifetime value
        
        # Calculate ROI
        roi = []
        for i in range(len(strategies)):
            # ROI = (Benefits - Costs) / Costs
            # Benefits = Retained customers * Customer value
            benefit = retention_rates[i] * customer_values[i]
            r = (benefit - costs[i]) / costs[i]
            roi.append(r)
        
        # Create a dataframe for visualization
        roi_df = pd.DataFrame({
            'Strategy': strategies,
            'Implementation Cost': costs,
            'Retention Rate': retention_rates,
            'ROI': roi
        })
        
        # Sort by ROI
        roi_df = roi_df.sort_values('ROI', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            roi_df,
            x='Strategy',
            y='ROI',
            color='Retention Rate',
            text=[f"{r:.1%}" for r in roi_df['ROI']],
            color_continuous_scale="Blues",
            labels={"ROI": "Return on Investment"},
            title="ROI of Different Retention Strategies"
        )
        
        fig.update_layout(
            yaxis=dict(tickformat='.0%'),
            xaxis_title=""
        )
        
        st.plotly_chart(fig)
        
            
        with st.expander("View Budget Allocation"):
            strategy_budget = {
                "Loyalty Program": "35%",
                "Service Improvements": "25%",
                "Targeted Discounts": "20%",
                "Customer Support": "15%",
                "Analytics & Monitoring": "5%"
            }
            
            fig = px.pie(
                values=list(map(lambda x: float(x.strip('%')), strategy_budget.values())),
                names=strategy_budget.keys(),
                title="Recommended Budget Allocation",
                hole=0.4
            )
            
            st.plotly_chart(fig)