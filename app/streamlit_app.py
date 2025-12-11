"""
Credit Risk Probability Dashboard
A modern Streamlit application for credit risk prediction.
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import streamlit as st
import pandas as pd

from predict import load_model, load_model_metadata, predict_single, get_feature_columns

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Modern Fintech Design
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    /* Import Google Font - Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Card styling */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
    }
    
    .metric-card h3 {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0 0 0.5rem 0;
    }
    
    .metric-card .value {
        color: #1e293b;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Risk prediction cards */
    .risk-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #10b981;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
    }
    
    .risk-card h2 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    
    .risk-low h2 {
        color: #065f46;
    }
    
    .risk-high h2 {
        color: #991b1b;
    }
    
    .risk-card .probability {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .risk-low .probability {
        color: #059669;
    }
    
    .risk-high .probability {
        color: #dc2626;
    }
    
    .risk-card .description {
        font-size: 0.95rem;
        color: #475569;
        margin: 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #0f172a;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 0.5rem 0.75rem;
    }
    
    .stSlider > div > div > div {
        background-color: #3b82f6;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        color: #1e293b;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Feature guide cards */
    .feature-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #3b82f6;
    }
    
    .feature-card h4 {
        color: #1e293b;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0 0 0.25rem 0;
    }
    
    .feature-card p {
        color: #64748b;
        font-size: 0.85rem;
        margin: 0;
    }
    
    /* Metrics table */
    .metrics-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    }
    
    .metrics-table th {
        background: #1e293b;
        color: white;
        padding: 1rem;
        font-weight: 600;
        text-align: left;
    }
    
    .metrics-table td {
        padding: 1rem;
        border-bottom: 1px solid #e2e8f0;
        background: white;
    }
    
    .metrics-table tr:last-child td {
        border-bottom: none;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

@st.cache_resource
def get_model():
    """Load and cache the model."""
    return load_model()


@st.cache_resource
def get_metadata():
    """Load and cache model metadata."""
    return load_model_metadata()


def render_header():
    """Render the main header."""
    st.markdown("""
        <div class="main-header">
            <h1>🏦 Credit Risk Probability Dashboard</h1>
            <p>Predict customer credit risk using machine learning</p>
        </div>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, col):
    """Render a metric card."""
    with col:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{title}</h3>
                <p class="value">{value}</p>
            </div>
        """, unsafe_allow_html=True)


def render_risk_prediction(is_high_risk: int, probability: float):
    """Render the risk prediction card."""
    if is_high_risk == 0:
        risk_class = "risk-low"
        risk_label = "Low Risk"
        risk_icon = "✅"
        description = "This customer shows healthy transaction patterns and is unlikely to default."
    else:
        risk_class = "risk-high"
        risk_label = "High Risk"
        risk_icon = "⚠️"
        description = "This customer shows patterns associated with higher default probability. Consider additional verification."
    
    st.markdown(f"""
        <div class="risk-card {risk_class}">
            <h2>{risk_icon} {risk_label}</h2>
            <p class="probability">{probability:.1%}</p>
            <p class="description">{description}</p>
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------

def render_overview_tab():
    """Render the Overview tab."""
    st.markdown('<p class="section-header">📊 Project Overview</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Credit Risk Probability Model** is a machine learning system that predicts 
        whether a customer is likely to be a high-risk borrower based on their 
        transaction behavior.
        
        #### How It Works
        
        1. **Data Collection**: Customer transaction data is aggregated into behavioral features.
        2. **Feature Engineering**: RFM (Recency, Frequency, Monetary) metrics and transaction patterns are computed.
        3. **Risk Scoring**: A trained Random Forest model predicts the probability of high risk.
        
        #### Key Features Used
        
        - **Transaction patterns**: Count, amounts, timing
        - **RFM metrics**: Recency, frequency, monetary value
        - **Channel & product usage**: Distribution across categories
        """)
    
    with col2:
        # Load metadata
        try:
            metadata = get_metadata()
            model_name = metadata.get("model_name", "Unknown")
            metrics = metadata.get("metrics", {})
            
            st.markdown('<p class="section-header">🎯 Model Performance</p>', unsafe_allow_html=True)
            
            st.metric("Model", model_name)
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            st.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
            
        except Exception as e:
            st.warning(f"Could not load model metadata: {e}")
    
    st.markdown("---")
    
    st.markdown('<p class="section-header">🚀 Quick Start</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Step 1</h3>
            <p class="value">📝 Input</p>
            <p style="color: #64748b; font-size: 0.9rem;">Enter customer features in the Prediction tab</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Step 2</h3>
            <p class="value">🔮 Predict</p>
            <p style="color: #64748b; font-size: 0.9rem;">Click the predict button to get risk score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Step 3</h3>
            <p class="value">📊 Analyze</p>
            <p style="color: #64748b; font-size: 0.9rem;">Review the risk assessment and probability</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Single Prediction
# ---------------------------------------------------------------------------

def render_prediction_tab():
    """Render the Single Prediction tab."""
    st.markdown('<p class="section-header">🔮 Customer Risk Prediction</p>', unsafe_allow_html=True)
    
    # Create two columns: inputs on left, results on right
    input_col, result_col = st.columns([1.5, 1])
    
    with input_col:
        st.markdown("#### Enter Customer Features")
        
        # Transaction Activity
        with st.expander("📊 Transaction Activity", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                txn_count = st.number_input("Transaction Count", min_value=1, value=10, step=1)
                frequency = st.number_input("Frequency (RFM)", min_value=1, value=10, step=1)
            with col2:
                recency_days = st.number_input("Recency (days)", min_value=0, value=15, step=1)
                n_credits = st.number_input("Credit Transactions", min_value=0, value=3, step=1)
            with col3:
                n_debits = st.number_input("Debit Transactions", min_value=0, value=7, step=1)
        
        # Monetary Values
        with st.expander("💰 Monetary Values", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                total_amount = st.number_input("Total Amount", value=5000.0, step=100.0)
                avg_amount = st.number_input("Avg Amount", value=500.0, step=10.0)
                std_amount = st.number_input("Std Amount", value=200.0, step=10.0)
            with col2:
                min_amount = st.number_input("Min Amount", value=-100.0, step=10.0)
                max_amount = st.number_input("Max Amount", value=1000.0, step=10.0)
                net_amount = st.number_input("Net Amount", value=4500.0, step=100.0)
            with col3:
                total_value = st.number_input("Total Value", value=5500.0, step=100.0)
                avg_value = st.number_input("Avg Value", value=550.0, step=10.0)
                monetary = st.number_input("Monetary (RFM)", value=5500.0, step=100.0)
        
        # Time-based Features
        with st.expander("🕐 Time-based Features", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_txn_hour = st.slider("Avg Transaction Hour", 0.0, 23.0, 14.5, 0.5)
            with col2:
                std_txn_hour = st.slider("Std Transaction Hour", 0.0, 12.0, 3.2, 0.1)
            with col3:
                weekend_txn_ratio = st.slider("Weekend Ratio", 0.0, 1.0, 0.2, 0.05)
        
        # Category Ratios
        with st.expander("📂 Category Ratios", expanded=False):
            st.markdown("**Product Category**")
            col1, col2, col3 = st.columns(3)
            with col1:
                pc_financial = st.slider("Financial Services", 0.0, 1.0, 0.5, 0.05)
            with col2:
                pc_airtime = st.slider("Airtime", 0.0, 1.0, 0.3, 0.05)
            with col3:
                pc_utility = st.slider("Utility Bill", 0.0, 1.0, 0.2, 0.05)
            
            st.markdown("**Channel**")
            col1, col2, col3 = st.columns(3)
            with col1:
                ch_3 = st.slider("Channel 3", 0.0, 1.0, 0.6, 0.05)
            with col2:
                ch_2 = st.slider("Channel 2", 0.0, 1.0, 0.3, 0.05)
            with col3:
                ch_5 = st.slider("Channel 5", 0.0, 1.0, 0.1, 0.05)
            
            st.markdown("**Provider**")
            col1, col2, col3 = st.columns(3)
            with col1:
                prov_4 = st.slider("Provider 4", 0.0, 1.0, 0.4, 0.05)
            with col2:
                prov_6 = st.slider("Provider 6", 0.0, 1.0, 0.3, 0.05)
            with col3:
                prov_5 = st.slider("Provider 5", 0.0, 1.0, 0.3, 0.05)
        
        # Predict button
        st.markdown("")
        predict_button = st.button("🔮 Predict Risk", use_container_width=True)
    
    with result_col:
        st.markdown("#### Prediction Result")
        
        if predict_button:
            # Build features dict
            features = {
                "txn_count": txn_count,
                "total_amount": total_amount,
                "avg_amount": avg_amount,
                "std_amount": std_amount,
                "min_amount": min_amount,
                "max_amount": max_amount,
                "total_value": total_value,
                "avg_value": avg_value,
                "avg_txn_hour": avg_txn_hour,
                "std_txn_hour": std_txn_hour,
                "weekend_txn_ratio": weekend_txn_ratio,
                "net_amount": net_amount,
                "n_credits": n_credits,
                "n_debits": n_debits,
                "productcategory_financial_services_ratio": pc_financial,
                "productcategory_airtime_ratio": pc_airtime,
                "productcategory_utility_bill_ratio": pc_utility,
                "channelid_channelid_3_ratio": ch_3,
                "channelid_channelid_2_ratio": ch_2,
                "channelid_channelid_5_ratio": ch_5,
                "providerid_providerid_4_ratio": prov_4,
                "providerid_providerid_6_ratio": prov_6,
                "providerid_providerid_5_ratio": prov_5,
                "recency_days": recency_days,
                "frequency": frequency,
                "monetary": monetary,
            }
            
            try:
                # Make prediction
                result = predict_single(features)
                
                # Render result
                render_risk_prediction(
                    is_high_risk=result["is_high_risk"],
                    probability=result["probability_high_risk"]
                )
                
                st.markdown("")
                
                # Additional details
                with st.expander("📋 Prediction Details"):
                    st.json({
                        "is_high_risk": result["is_high_risk"],
                        "probability_high_risk": f"{result['probability_high_risk']:.4f}",
                        "model_name": result["model_name"],
                        "model_version": result["model_version"],
                    })
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Make sure the model is trained and saved. Run: `python scripts/save_best_model.py`")
        
        else:
            # Placeholder
            st.markdown("""
            <div style="
                background: #f1f5f9;
                border-radius: 16px;
                padding: 3rem 2rem;
                text-align: center;
                border: 2px dashed #cbd5e1;
            ">
                <p style="font-size: 3rem; margin: 0;">🔮</p>
                <p style="color: #64748b; font-size: 1.1rem; margin: 1rem 0 0 0;">
                    Fill in customer features and click <strong>Predict Risk</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Model Performance
# ---------------------------------------------------------------------------

def render_performance_tab():
    """Render the Model Performance tab."""
    st.markdown('<p class="section-header">📈 Model Performance Metrics</p>', unsafe_allow_html=True)
    
    try:
        metadata = get_metadata()
        model_name = metadata.get("model_name", "Unknown")
        metrics = metadata.get("metrics", {})
        
        # Model info
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h3>Best Model</h3>
                <p class="value" style="color: #3b82f6;">{model_name}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            The model was selected based on **ROC-AUC** score after hyperparameter 
            tuning with 5-fold cross-validation using GridSearchCV.
            """)
        
        st.markdown("")
        
        # Metrics grid
        st.markdown("#### Evaluation Metrics (Test Set)")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>ROC-AUC</h3>
                <p class="value">{metrics.get('roc_auc', 0):.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Accuracy</h3>
                <p class="value">{metrics.get('accuracy', 0):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Precision</h3>
                <p class="value">{metrics.get('precision', 0):.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Recall</h3>
                <p class="value">{metrics.get('recall', 0):.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h3>F1 Score</h3>
                <p class="value">{metrics.get('f1', 0):.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("---")
        
        # Metric explanations
        st.markdown("#### Metric Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🎯 ROC-AUC (Receiver Operating Characteristic - Area Under Curve)</h4>
                <p>Measures the model's ability to distinguish between classes. 
                1.0 = perfect, 0.5 = random guessing. Our model achieves near-perfect discrimination.</p>
            </div>
            
            <div class="feature-card">
                <h4>✅ Accuracy</h4>
                <p>Percentage of correct predictions out of all predictions. 
                High accuracy indicates reliable overall performance.</p>
            </div>
            
            <div class="feature-card">
                <h4>🔍 Precision</h4>
                <p>Of all predicted high-risk customers, how many are actually high-risk? 
                High precision means fewer false alarms.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>📡 Recall (Sensitivity)</h4>
                <p>Of all actual high-risk customers, how many did we correctly identify? 
                High recall means we catch most risky customers.</p>
            </div>
            
            <div class="feature-card">
                <h4>⚖️ F1 Score</h4>
                <p>Harmonic mean of precision and recall. Balances the trade-off between 
                catching all risky customers and avoiding false positives.</p>
            </div>
            
            <div class="feature-card">
                <h4>📊 Model Selection</h4>
                <p>Random Forest was chosen as the best model based on cross-validation 
                performance. It handles complex feature interactions well.</p>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Could not load model performance data: {e}")


# ---------------------------------------------------------------------------
# Tab: Feature Guide
# ---------------------------------------------------------------------------

def render_feature_guide_tab():
    """Render the Feature Guide tab."""
    st.markdown('<p class="section-header">📚 Feature Guide</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This guide explains each feature used by the credit risk model. Understanding 
    these features helps interpret predictions and prepare input data.
    """)
    
    st.markdown("")
    
    # RFM Features
    st.markdown("#### 🎯 RFM (Recency, Frequency, Monetary) Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>recency_days</h4>
            <p>Days since the customer's last transaction. Lower values indicate recent activity (typically good).</p>
        </div>
        
        <div class="feature-card">
            <h4>frequency</h4>
            <p>Total number of transactions. Higher frequency often indicates engaged, reliable customers.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>monetary</h4>
            <p>Total monetary value of all transactions. Higher values suggest more significant customer relationships.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Transaction Features
    st.markdown("#### 💳 Transaction Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>txn_count</h4>
            <p>Number of transactions in the analysis period.</p>
        </div>
        
        <div class="feature-card">
            <h4>total_amount / avg_amount / std_amount</h4>
            <p>Sum, average, and standard deviation of transaction amounts. Captures spending patterns and consistency.</p>
        </div>
        
        <div class="feature-card">
            <h4>min_amount / max_amount</h4>
            <p>Range of transaction amounts. Wide ranges may indicate varied transaction behavior.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>n_credits / n_debits</h4>
            <p>Count of credit (incoming) vs debit (outgoing) transactions. Balance indicates cash flow direction.</p>
        </div>
        
        <div class="feature-card">
            <h4>net_amount</h4>
            <p>Total credits minus debits. Positive = net inflow, negative = net outflow.</p>
        </div>
        
        <div class="feature-card">
            <h4>total_value / avg_value</h4>
            <p>Absolute transaction values (regardless of credit/debit direction).</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Time Features
    st.markdown("#### 🕐 Time-based Features")
    
    st.markdown("""
    <div class="feature-card">
        <h4>avg_txn_hour / std_txn_hour</h4>
        <p>Average and variation in transaction timing (0-23 hours). Unusual timing patterns may indicate risk.</p>
    </div>
    
    <div class="feature-card">
        <h4>weekend_txn_ratio</h4>
        <p>Proportion of transactions on weekends vs weekdays. Different customer segments have different patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Category Ratios
    st.markdown("#### 📂 Category Ratio Features")
    
    st.markdown("""
    These features represent the proportion of transactions in each category. 
    Values range from 0 to 1, where 1 means all transactions are in that category.
    
    - **Product Categories**: financial_services, airtime, utility_bill
    - **Channels**: Different transaction channels (mobile, web, etc.)
    - **Providers**: Different service providers
    
    Transaction diversification across categories may indicate different risk profiles.
    """)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    """Main application entry point."""
    # Render header
    render_header()
    
    # Load model on startup
    try:
        get_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
        st.warning("""
        ⚠️ **Model not found!** The prediction feature requires a trained model.
        
        Run this command to train and save the model:
        ```bash
        python scripts/save_best_model.py
        ```
        """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Overview",
        "🔮 Prediction",
        "📈 Performance",
        "📚 Feature Guide"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        if model_loaded:
            render_prediction_tab()
        else:
            st.info("Load the model first to enable predictions.")
    
    with tab3:
        render_performance_tab()
    
    with tab4:
        render_feature_guide_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">
        Credit Risk Probability Model • Built with Streamlit • 
        <a href="https://github.com" style="color: #3b82f6;">GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
