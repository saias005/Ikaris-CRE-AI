"""
CBRE ML Models - Practical Predictions from CSV Data
=====================================================

Here are the standard ML models we'll implement for different types of predictions:
"""

# 1. REGRESSION MODELS (Predicting continuous values)
# ====================================================

## A. Property Value Prediction
"""
Model: Random Forest Regressor or XGBoost
Input Features from CSV:
- total_sqft, building_age, occupancy_rate
- noi_annual, cap_rate, market, property_type
- energy_star_score, building_class

Output: Predicted property value ($)

Use Case: "What should this property be worth?"
         "Is this property overvalued or undervalued?"
"""

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def train_property_value_model(df):
    features = ['total_sqft', 'building_age', 'occupancy_rate', 
                'noi_annual', 'cap_rate', 'energy_star_score']
    
    X = df[features]
    y = df['property_value']
    
    # Random Forest - good for non-linear relationships, handles outliers well
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    # OR XGBoost - often better accuracy
    # model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    
    model.fit(X, y)
    return model


## B. Future Maintenance Cost Prediction
"""
Model: Gradient Boosting Regressor
Input Features:
- building_age, property_type, total_sqft
- current maintenance_annual, maintenance_risk_score
- historical maintenance costs (from historical_metrics.csv)

Output: Predicted maintenance cost for next year ($)

Use Case: "Which properties will have high maintenance costs next year?"
         "Budget forecast for maintenance expenses"
"""

from sklearn.ensemble import GradientBoostingRegressor

def train_maintenance_predictor(df, historical_df):
    # Aggregate historical maintenance costs
    hist_maintenance = historical_df.groupby('property_id')['maintenance_cost'].agg(['mean', 'std', 'max'])
    df = df.merge(hist_maintenance, on='property_id')
    
    features = ['building_age', 'total_sqft', 'maintenance_annual', 
                'maintenance_risk_score', 'maintenance_cost_mean']
    
    # Create target: next year's maintenance (simulate with increase based on age)
    df['future_maintenance'] = df['maintenance_annual'] * (1 + df['building_age']/50)
    
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(df[features], df['future_maintenance'])
    return model


## C. Rent Growth Prediction
"""
Model: Linear Regression with polynomial features
Input Features:
- market_rent_growth_yoy, occupancy_rate
- market, property_type, building_class
- Historical rent trends

Output: Predicted rent per sqft next year

Use Case: "Which properties can support rent increases?"
         "Forecast rental income for next year"
"""

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

def train_rent_predictor(df):
    features = ['base_rent_psf', 'occupancy_rate', 'market_rent_growth_yoy']
    
    # Polynomial features for non-linear patterns
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(df[features])
    
    # Ridge regression (handles multicollinearity)
    model = Ridge(alpha=1.0)
    model.fit(X_poly, df['effective_rent_psf'])
    return model, poly


# 2. CLASSIFICATION MODELS (Predicting categories/probabilities)
# ===============================================================

## A. Lease Renewal Risk Classification
"""
Model: Random Forest Classifier
Input Features:
- walt_years (weighted avg lease term)
- tenant_risk_score, renewal_probability
- occupancy_rate, market_vacancy_rate
- tenant payment_history_score

Output: Risk category (High/Medium/Low) or probability of non-renewal

Use Case: "Which leases are at risk of not renewing?"
         "Identify properties needing tenant retention efforts"
"""

from sklearn.ensemble import RandomForestClassifier

def train_lease_risk_model(df, tenants_df):
    # Aggregate tenant data
    tenant_agg = tenants_df.groupby('property_id').agg({
        'renewal_probability': 'mean',
        'payment_history_score': 'mean'
    })
    df = df.merge(tenant_agg, on='property_id')
    
    # Create risk labels
    df['high_risk'] = ((df['walt_years'] < 1) & 
                       (df['renewal_probability'] < 0.6)).astype(int)
    
    features = ['walt_years', 'occupancy_rate', 'tenant_risk_score',
                'renewal_probability', 'payment_history_score']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=8, 
                                  class_weight='balanced')  # Handle imbalanced data
    model.fit(df[features], df['high_risk'])
    return model


## B. Energy Efficiency Opportunity Identifier
"""
Model: Logistic Regression
Input Features:
- energy_star_score, energy_cost_psf
- building_age, leed_certified
- property_type, total_sqft

Output: Probability of achieving significant energy savings

Use Case: "Which properties have the best energy optimization opportunities?"
         "Identify candidates for energy upgrades"
"""

from sklearn.linear_model import LogisticRegression

def train_energy_opportunity_model(df):
    # Create target: properties with high savings potential
    df['high_energy_potential'] = (
        (df['energy_star_score'] < 75) & 
        (df['energy_cost_psf'] > df['energy_cost_psf'].median())
    ).astype(int)
    
    features = ['energy_star_score', 'energy_cost_psf', 'building_age',
                'leed_certified', 'total_sqft']
    
    model = LogisticRegression(class_weight='balanced')
    model.fit(df[features], df['high_energy_potential'])
    return model


# 3. CLUSTERING MODELS (Finding patterns/groups)
# ================================================

## A. Property Portfolio Segmentation
"""
Model: K-Means Clustering
Input Features:
- All financial and operational metrics

Output: Property segments/clusters

Use Case: "Group similar properties for portfolio strategy"
         "Identify peer properties for comparison"
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_property_segments(df):
    features = ['occupancy_rate', 'noi_per_sqft', 'cap_rate', 
                'energy_cost_psf', 'maintenance_risk_score']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # Find optimal clusters (typically 4-6 for commercial real estate)
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['segment'] = kmeans.fit_predict(X_scaled)
    
    return kmeans, scaler


# 4. TIME SERIES MODELS (Predicting trends)
# ==========================================

## A. Occupancy Rate Forecasting
"""
Model: ARIMA or Prophet (or simple moving average + regression)
Input: Historical occupancy rates from historical_metrics.csv

Output: Forecasted occupancy for next 6 months

Use Case: "Predict occupancy trends for next quarter"
         "Which properties will likely have vacancies?"
"""

from sklearn.linear_model import LinearRegression

def forecast_occupancy(historical_df, property_id):
    # Get property's historical data
    prop_history = historical_df[historical_df['property_id'] == property_id].copy()
    prop_history['month_num'] = range(len(prop_history))
    
    # Simple trend model
    X = prop_history[['month_num']]
    y = prop_history['occupancy_rate']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast next 6 months
    future_months = np.array([[len(prop_history) + i] for i in range(6)])
    forecast = model.predict(future_months)
    
    return forecast


# 5. ANOMALY DETECTION (Finding outliers)
# =========================================

## A. Expense Anomaly Detection
"""
Model: Isolation Forest
Input: Operating expenses across all properties

Output: Properties with unusual expense patterns

Use Case: "Which properties have abnormal operating costs?"
         "Identify potential billing errors or maintenance issues"
"""

from sklearn.ensemble import IsolationForest

def detect_expense_anomalies(df):
    features = ['opex_per_sqft', 'energy_cost_psf', 'maintenance_annual',
                'property_tax_annual', 'insurance_annual']
    
    # Normalize by square footage
    X = df[features].values
    
    # Isolation Forest is good for anomaly detection
    model = IsolationForest(contamination=0.1)  # Flag 10% as anomalies
    df['is_anomaly'] = model.fit_predict(X)
    
    return model


# 6. PRACTICAL DECISION TREE FOR RECOMMENDATIONS
# ================================================

"""
Model: Decision Tree Regressor/Classifier
Purpose: Explainable AI - show WHY a recommendation is made

Use Case: "Why is this property considered high risk?"
         "What factors drive this valuation?"
"""

from sklearn.tree import DecisionTreeRegressor, export_text

def create_explainable_model(df, target='property_value'):
    features = ['occupancy_rate', 'noi_annual', 'building_age', 'cap_rate']
    
    model = DecisionTreeRegressor(max_depth=5)  # Keep shallow for interpretability
    model.fit(df[features], df[target])
    
    # Get human-readable rules
    tree_rules = export_text(model, feature_names=features)
    
    return model, tree_rules


# EXAMPLE USAGE IN YOUR QUERY HANDLER
# ====================================

class MLPredictionHandler:
    def __init__(self, properties_df, tenants_df, historical_df):
        self.properties = properties_df
        self.tenants = tenants_df
        self.historical = historical_df
        
        # Train models on initialization
        self.models = {}
        self.models['value'] = train_property_value_model(properties_df)
        self.models['maintenance'] = train_maintenance_predictor(properties_df, historical_df)
        self.models['lease_risk'] = train_lease_risk_model(properties_df, tenants_df)
        
    def predict_for_query(self, query):
        """Route query to appropriate model"""
        
        query_lower = query.lower()
        
        if 'value' in query_lower or 'worth' in query_lower:
            return self.predict_values()
        
        elif 'maintenance' in query_lower or 'repair' in query_lower:
            return self.predict_maintenance()
        
        elif 'lease' in query_lower or 'renewal' in query_lower:
            return self.predict_lease_risk()
        
        elif 'energy' in query_lower or 'efficiency' in query_lower:
            return self.identify_energy_opportunities()
        
        else:
            return "Query doesn't require ML prediction. Using standard RAG search."
    
    def predict_values(self):
        """Predict which properties are under/over valued"""
        
        # Get predictions
        predicted_values = self.models['value'].predict(self.properties[features])
        
        # Compare to actual
        self.properties['predicted_value'] = predicted_values
        self.properties['value_diff'] = predicted_values - self.properties['property_value']
        self.properties['value_diff_pct'] = (self.properties['value_diff'] / 
                                              self.properties['property_value'] * 100)
        
        # Find opportunities
        undervalued = self.properties[self.properties['value_diff_pct'] > 10]
        overvalued = self.properties[self.properties['value_diff_pct'] < -10]
        
        response = f"PROPERTY VALUE ANALYSIS:\n\n"
        response += f"Potentially Undervalued Properties (Buy/Hold opportunities):\n"
        for _, prop in undervalued.nlargest(5, 'value_diff').iterrows():
            response += f"- {prop['property_name']}: Currently ${prop['property_value']:,.0f}, "
            response += f"Predicted ${prop['predicted_value']:,.0f} "
            response += f"(+{prop['value_diff_pct']:.1f}% potential)\n"
        
        return response


# SUMMARY OF ML MODELS FOR CBRE DATA:
"""
1. REGRESSION (Predict Numbers):
   - Property values
   - Future maintenance costs  
   - Rent growth rates
   - Energy costs

2. CLASSIFICATION (Predict Categories):
   - Lease renewal risk (High/Med/Low)
   - Energy efficiency opportunities (Yes/No)
   - Tenant default risk

3. CLUSTERING (Find Patterns):
   - Property segmentation
   - Market groupings
   - Tenant clustering

4. TIME SERIES (Predict Trends):
   - Occupancy forecasting
   - NOI projections
   - Market trend analysis

5. ANOMALY DETECTION (Find Outliers):
   - Expense anomalies
   - Unusual tenant behavior
   - Market outliers

These models use standard scikit-learn algorithms that are:
- Fast to train (seconds on 500 properties)
- Interpretable (can explain predictions)
- Proven in real estate analytics
- Don't require deep learning or GPUs
"""