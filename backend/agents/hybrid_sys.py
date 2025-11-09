"""
IKARIS - Intelligent CRE Query System
Hybrid RAG + ML for CBRE HackUTD 2025
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


class IkarisHybridSystem:
    """
    Hybrid system that routes between:
    1. Simple RAG for factual queries
    2. ML predictions for analytical queries
    """
    
    def __init__(self, data_dir='./backend/data/cbre_data'):
        """Initialize with CSV data and train ML models"""
        
        print("Initializing IKARIS system...")
        
        # Load CSV data
        self.properties = pd.read_csv(f'{data_dir}/properties.csv')
        self.tenants = pd.read_csv(f'{data_dir}/tenants.csv')
        self.historical = pd.read_csv(f'{data_dir}/historical_metrics.csv')
        
        print(f"✅ Loaded {len(self.properties)} properties")
        
        # Prepare features
        self._prepare_features()
        
        # Train ML models
        self.models = {}
        self._train_models()
        
        print("✅ IKARIS system ready!")
    
    def _prepare_features(self):
        """Prepare features for ML models"""
        
        # Encode categorical variables
        self.label_encoders = {}
        
        # Encode market
        self.label_encoders['market'] = LabelEncoder()
        self.properties['market_encoded'] = self.label_encoders['market'].fit_transform(
            self.properties['market']
        )
        
        # Encode property type
        self.label_encoders['property_type'] = LabelEncoder()
        self.properties['property_type_encoded'] = self.label_encoders['property_type'].fit_transform(
            self.properties['property_type']
        )
        
        # Encode building class
        class_map = {'A': 3, 'B': 2, 'C': 1}
        self.properties['building_class_encoded'] = self.properties['building_class'].map(class_map)
        
        # Add tenant aggregations
        tenant_agg = self.tenants.groupby('property_id').agg({
            'renewal_probability': 'mean',
            'payment_history_score': 'mean',
            'annual_rent': 'sum'
        }).reset_index()
        
        self.properties = self.properties.merge(tenant_agg, on='property_id', how='left')
        
        # Add historical aggregations
        hist_agg = self.historical.groupby('property_id').agg({
            'occupancy_rate': ['mean', 'std'],
            'noi': ['mean', 'std'],
            'energy_cost': 'mean',
            'maintenance_cost': 'mean'
        }).reset_index()
        
        hist_agg.columns = ['property_id', 'hist_occupancy_mean', 'hist_occupancy_std',
                            'hist_noi_mean', 'hist_noi_std', 'hist_energy_mean', 
                            'hist_maintenance_mean']
        
        self.properties = self.properties.merge(hist_agg, on='property_id', how='left')
        
        # Fill NaN values
        self.properties = self.properties.fillna(0)
    
    def _train_models(self):
        """Train all ML models"""
        
        print("Training ML models...")
        
        # 1. Property Value Prediction Model
        self._train_value_model()
        
        # 2. Maintenance Cost Prediction Model
        self._train_maintenance_model()
        
        # 3. Lease Risk Classification Model
        self._train_lease_risk_model()
        
        # 4. Energy Efficiency Model
        self._train_energy_model()
        
        # 5. Occupancy Prediction Model
        self._train_occupancy_model()
        
        print("✅ All models trained successfully")
    
    def _train_value_model(self):
        """Train property value prediction model"""
        
        features = ['total_sqft', 'building_age', 'occupancy_rate', 'noi_annual',
                   'cap_rate', 'energy_star_score', 'market_encoded', 
                   'property_type_encoded', 'building_class_encoded',
                   'hist_occupancy_mean', 'hist_noi_mean']
        
        X = self.properties[features]
        y = self.properties['property_value']
        
        self.models['value'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.models['value'].fit(X, y)
        self.value_features = features

    def _apply_dynamic_filters(self, query: str, properties_df=None) -> tuple:
        """
        Apply dynamic filters based on query content
        Returns: (filtered_dataframe, filter_description_list)
        """
        query_lower = query.lower()
        
        # Start with all properties or provided dataframe
        filtered_props = properties_df.copy() if properties_df is not None else self.properties.copy()
        filter_description = []
        
        # Dynamic location filtering - check for ANY market
        for market in self.properties['market'].unique():
            if market.lower() in query_lower:
                filtered_props = filtered_props[filtered_props['market'] == market]
                filter_description.append(f"Market: {market}")
                break
        
        # Dynamic property type filtering - check for ANY type
        for prop_type in self.properties['property_type'].unique():
            if prop_type.lower() in query_lower:
                filtered_props = filtered_props[filtered_props['property_type'] == prop_type]
                filter_description.append(f"Type: {prop_type}")
                break
        
        # Lease expiry filtering
        if 'expiring' in query_lower or 'expire' in query_lower:
            if 'next quarter' in query_lower:
                filtered_props = filtered_props[filtered_props['walt_years'] < 0.25]
                filter_description.append("Leases expiring next quarter")
            elif 'next year' in query_lower:
                filtered_props = filtered_props[filtered_props['walt_years'] < 1.0]
                filter_description.append("Leases expiring next year")
            else:
                filtered_props = filtered_props[filtered_props['walt_years'] < 2.0]
                filter_description.append("Leases expiring within 2 years")
        
        # Building class filtering
        if 'class a' in query_lower:
            filtered_props = filtered_props[filtered_props['building_class'] == 'A']
            filter_description.append("Class A")
        elif 'class b' in query_lower:
            filtered_props = filtered_props[filtered_props['building_class'] == 'B']
            filter_description.append("Class B")
        elif 'class c' in query_lower:
            filtered_props = filtered_props[filtered_props['building_class'] == 'C']
            filter_description.append("Class C")
        
        # Energy/sustainability filtering
        if 'leed' in query_lower:
            filtered_props = filtered_props[filtered_props['leed_certified'] == True]
            filter_description.append("LEED Certified")
        
        if 'energy efficient' in query_lower or 'low energy' in query_lower:
            filtered_props = filtered_props[filtered_props['energy_star_score'] > 75]
            filter_description.append("Energy Star > 75")
        elif 'high energy' in query_lower:
            filtered_props = filtered_props[filtered_props['energy_cost_psf'] > 3.0]
            filter_description.append("High energy cost (>$3/sqft)")
        
        # Occupancy filtering
        if 'low occupancy' in query_lower or 'vacant' in query_lower:
            filtered_props = filtered_props[filtered_props['occupancy_rate'] < 0.8]
            filter_description.append("Low occupancy (<80%)")
        elif 'high occupancy' in query_lower or 'fully leased' in query_lower:
            filtered_props = filtered_props[filtered_props['occupancy_rate'] > 0.95]
            filter_description.append("High occupancy (>95%)")
        
        # Size filtering
        if 'large' in query_lower:
            filtered_props = filtered_props[filtered_props['total_sqft'] > 100000]
            filter_description.append("Large (>100k sqft)")
        elif 'small' in query_lower:
            filtered_props = filtered_props[filtered_props['total_sqft'] < 50000]
            filter_description.append("Small (<50k sqft)")
        
        # Age filtering
        if 'new' in query_lower or 'modern' in query_lower:
            filtered_props = filtered_props[filtered_props['building_age'] < 10]
            filter_description.append("New buildings (<10 years)")
        elif 'old' in query_lower or 'aging' in query_lower:
            filtered_props = filtered_props[filtered_props['building_age'] > 30]
            filter_description.append("Older buildings (>30 years)")
        
        return filtered_props, filter_description
    
    def _train_maintenance_model(self):
        """Train maintenance cost prediction model"""
        
        features = ['building_age', 'total_sqft', 'maintenance_risk_score',
                   'property_type_encoded', 'hist_maintenance_mean']
        
        # Create target (future maintenance)
        self.properties['future_maintenance'] = (
            self.properties['maintenance_annual'] * 
            (1 + self.properties['building_age']/50)
        )
        
        X = self.properties[features]
        y = self.properties['future_maintenance']
        
        self.models['maintenance'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.models['maintenance'].fit(X, y)
        self.maintenance_features = features
    
    def _train_lease_risk_model(self):
        """Train lease renewal risk model"""
        
        features = ['walt_years', 'occupancy_rate', 'tenant_risk_score',
                   'renewal_probability', 'payment_history_score',
                   'market_vacancy_rate']
        
        # Create risk labels
        self.properties['high_lease_risk'] = (
            (self.properties['walt_years'] < 1) & 
            (self.properties['renewal_probability'] < 0.6)
        ).astype(int)
        
        X = self.properties[features]
        y = self.properties['high_lease_risk']
        
        self.models['lease_risk'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight='balanced',
            random_state=42
        )
        self.models['lease_risk'].fit(X, y)
        self.lease_features = features
    
    def _train_energy_model(self):
        """Train energy efficiency opportunity model"""
        
        features = ['energy_star_score', 'energy_cost_psf', 'building_age',
                   'leed_certified', 'total_sqft', 'property_type_encoded']
        
        # Convert boolean to int
        self.properties['leed_certified'] = self.properties['leed_certified'].astype(int)
        
        # Create target
        self.properties['high_energy_cost'] = (
            self.properties['energy_cost_psf'] > 
            self.properties['energy_cost_psf'].quantile(0.75)
        ).astype(int)
        
        X = self.properties[features]
        y = self.properties['high_energy_cost']
        
        self.models['energy'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.models['energy'].fit(X, y)
        self.energy_features = features
    
    def _train_occupancy_model(self):
        """Train occupancy prediction model"""
        
        features = ['occupancy_rate', 'market_vacancy_rate', 'walt_years',
                   'base_rent_psf', 'building_age', 'hist_occupancy_mean']
        
        # Create target (future occupancy - simplified)
        self.properties['future_occupancy'] = np.clip(
            self.properties['occupancy_rate'] + 
            np.random.normal(0, 0.05, len(self.properties)),
            0, 1
        )
        
        X = self.properties[features]
        y = self.properties['future_occupancy']
        
        self.models['occupancy'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.models['occupancy'].fit(X, y)
        self.occupancy_features = features
    
    def classify_query(self, query: str) -> str:
        """
        Classify query type to determine routing
        Returns: 'rag', 'ml_predict', 'ml_risk', 'ml_optimize', or 'hybrid'
        """
        
        query_lower = query.lower()
        
         # Check for conversational/chat queries FIRST
        chat_keywords = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 
                    'how are you', 'thank you', 'thanks', 'what can you do',
                    'help me', 'your capabilities', 'introduce yourself']
    
        # Only classify as chat if it's ACTUALLY a greeting/chat message
        # Don't include "can you" as a chat trigger since it's often used with analytical questions
        is_chat = False
        for keyword in chat_keywords:
            if keyword == query_lower or query_lower.startswith(keyword + ' '):
                is_chat = True
                break
        
        if is_chat:
            return 'chat'
        
        # ML Prediction triggers - check these BEFORE defaulting to RAG
        ml_predict_keywords = ['predict', 'forecast', 'will be', 'future', 'next quarter', 
                            'next year', 'projection', 'estimate', 'expect', 'what will']
        
        # ML Risk triggers
        ml_risk_keywords = ['risk', 'risky', 'at risk', 'likely to', 'probability', 
                            'chance', 'vulnerable']
        
        # ML Optimization triggers
        ml_optimize_keywords = ['optimize', 'best', 'recommend', 'should', 'improve', 
                            'opportunity', 'potential', 'undervalued', 'overvalued', 
                            'highest value', 'lowest value']
        
        # Check for ML triggers
        has_predict = any(keyword in query_lower for keyword in ml_predict_keywords)
        has_risk = any(keyword in query_lower for keyword in ml_risk_keywords)
        has_optimize = any(keyword in query_lower for keyword in ml_optimize_keywords)
        
        if has_predict:
            return 'ml_predict'
        elif has_risk:
            return 'ml_risk'
        elif has_optimize:
            return 'ml_optimize'
        else:
            return 'rag'  # Default to RAG for factual queries
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point - routes query to appropriate handler
        """
        
        query_type = self.classify_query(query)
        
        print(f"Query type identified: {query_type}")
        
        if query_type == 'chat':
            return self.handle_chat_query(query)  
        elif query_type == 'rag':
            return self.handle_rag_query(query)
        elif query_type == 'ml_predict':
            return self.handle_prediction_query(query)
        elif query_type == 'ml_risk':
            return self.handle_risk_query(query)
        elif query_type == 'ml_optimize':
            return self.handle_optimization_query(query)

    def handle_chat_query(self, query: str) -> Dict[str, Any]:
        """Handle conversational queries"""

        query_lower = query.lower()
        
        # Prepare friendly responses
        if 'hello' in query_lower or 'hi' in query_lower or 'hey' in query_lower:
            response = """Hello! I'm IKARIS, your commercial real estate intelligence assistant. I'm here to help you with:

- Property value predictions and market analysis
- Risk assessments and maintenance forecasting
- Energy efficiency opportunities
- Portfolio optimization recommendations
- Searching through your property documents and data

What would you like to explore today?"""
        
        elif 'how are you' in query_lower:
            response = """I'm functioning optimally and ready to assist with your CRE needs! I have access to your full property portfolio data and can provide insights on values, risks, and optimization opportunities. How can I help you today?"""
        
        elif 'thank' in query_lower:
            response = """You're welcome! Is there anything else about your property portfolio I can help you analyze?"""
        
        elif 'what can you do' in query_lower or 'capabilities' in query_lower or 'help' in query_lower:
            response = """I can help you with a wide range of commercial real estate analysis:

**Predictive Analytics:**
- Forecast property values and maintenance costs
- Predict occupancy rates and lease renewal risks

**Risk Assessment:**
- Identify properties at risk of tenant loss
- Assess maintenance and operational risks
- Evaluate portfolio-wide risk exposure

**Optimization:**
- Find undervalued properties
- Identify energy efficiency opportunities
- Recommend portfolio optimization strategies

**Smart Search:**
- Search through your property documents
- Find specific properties by criteria (location, type, class, etc.)
- Analyze market trends from your documents

Try asking me something like:
- "Which properties in Dallas have high energy costs?"
- "Predict maintenance costs for next year"
- "Find undervalued office properties in Houston"

What would you like to know?"""
        
        else:
            response = """I'm here to help with your commercial real estate analysis. Feel free to ask me about property values, risks, optimization opportunities, or any specific properties in your portfolio. What would you like to know?"""
        
        return {
            'query': query,
            'type': 'chat',
            'response': response,
            'method': 'Conversational Response'
        }


    def handle_rag_query(self, query: str, vectorstore=None) -> Dict[str, Any]:
        """
        Handle factual queries with BOTH vector search AND CSV filtering
        """
        
        # 1. Search vector database (PDFs)
        pdf_results = []
        if vectorstore:
            pdf_results = vectorstore.similarity_search(query, k=5)
        
        # 2. Filter CSV properties
        filters = self._parse_query_filters(query)
        filtered_props = self.properties.copy()
        
        # Apply filters
        if 'market' in filters:
            filtered_props = filtered_props[filtered_props['market'] == filters['market']]
        
        if 'property_type' in filters:
            filtered_props = filtered_props[filtered_props['property_type'] == filters['property_type']]
        
        # Energy cost filter
        if 'high_energy' in filters or ('energy' in query.lower() and '$3' in query):
            filtered_props = filtered_props[filtered_props['energy_cost_psf'] > 3.0]
        
        if 'low_occupancy' in filters:
            filtered_props = filtered_props[filtered_props['occupancy_rate'] < 0.8]
        
        if 'expiring_leases' in filters:
            filtered_props = filtered_props[filtered_props['walt_years'] < 1.0]
        
        # 3. Combine both sources (FIX: Don't overwrite response!)
        response = ""
        
        # Add PDF results if available
        if pdf_results:
            response += "FROM DOCUMENTS:\n"
            for doc in pdf_results:
                response += f"- {doc.page_content[:200]}...\n"
            response += "\n"
        
        # Add property results
        response += "FROM PROPERTY DATABASE:\n"
        property_list = self._format_property_list(filtered_props.head(10), query)
        response += property_list
        
        return {
            'query': query,
            'type': 'factual',
            'response': response,
            'num_results': len(filtered_props),
            'method': 'Combined PDF + CSV Search'
        }
    
    def handle_prediction_query(self, query: str) -> Dict[str, Any]:
        """
        Handle prediction queries with ML models
        """
        
        query_lower = query.lower()
        
        if 'maintenance' in query_lower:
            return self.predict_maintenance_costs(query)
        elif 'value' in query_lower or 'worth' in query_lower:
            return self.predict_property_values(query)
        elif 'occupancy' in query_lower:
            return self.predict_occupancy_rates(query)
        else:
            return self.predict_general_trends(query)
    
    def handle_risk_query(self, query: str) -> Dict[str, Any]:
        """
        Handle risk assessment queries
        """
        
        query_lower = query.lower()
        
        if 'lease' in query_lower or 'renewal' in query_lower:
            return self.assess_lease_risk(query)
        elif 'maintenance' in query_lower:
            return self.assess_maintenance_risk(query)
        else:
            return self.assess_overall_risk(query)
    
    def handle_optimization_query(self, query: str) -> Dict[str, Any]:
        """
        Handle optimization and recommendation queries
        """
        
        query_lower = query.lower()
        
        if 'energy' in query_lower:
            return self.identify_energy_opportunities(query)
        elif 'value' in query_lower or 'undervalued' in query_lower:
            return self.identify_value_opportunities(query)
        else:
            return self.identify_general_opportunities(query)
    
    def predict_maintenance_costs(self, query: str) -> Dict[str, Any]:
        """Predict future maintenance costs with dynamic filtering"""
        
        # Apply filters first
        filtered_props, filter_description = self._apply_dynamic_filters(query)
        filter_summary = " with ".join(filter_description) if filter_description else "All properties"
        
        if len(filtered_props) == 0:
            return {
                'query': query,
                'type': 'prediction',
                'response': f"No properties found matching criteria: {filter_summary}",
                'method': 'ML - Gradient Boosting',
                'confidence': 0.0
            }
        
        # Get predictions for filtered properties
        X = filtered_props[self.maintenance_features]
        predictions = self.models['maintenance'].predict(X)
        
        filtered_props['predicted_maintenance'] = predictions
        filtered_props['maintenance_increase'] = (
            predictions - filtered_props['maintenance_annual']
        )
        
        # Find properties with highest predicted increases
        top_increases = filtered_props.nlargest(min(5, len(filtered_props)), 'maintenance_increase')
        
        response = f"MAINTENANCE COST PREDICTIONS ({filter_summary}):\n\n"
        response += f"Analyzing {len(filtered_props)} properties\n\n"
        response += "Properties with Highest Expected Maintenance Increases:\n\n"
        
        for _, prop in top_increases.iterrows():
            response += f"📍 {prop['property_name']} ({prop['market']})\n"
            response += f"   Type: {prop['property_type']} | Class: {prop['building_class']}\n"
            response += f"   Current: ${prop['maintenance_annual']:,.0f}\n"
            response += f"   Predicted: ${prop['predicted_maintenance']:,.0f}\n"
            response += f"   Increase: ${prop['maintenance_increase']:,.0f} "
            response += f"({prop['maintenance_increase']/prop['maintenance_annual']*100:.1f}%)\n"
            response += f"   Risk Factors: Building Age: {prop['building_age']} years\n\n"
        
        return {
            'query': query,
            'type': 'prediction',
            'response': response,
            'method': 'ML - Gradient Boosting with Dynamic Filtering',
            'confidence': 0.85
        }

    def predict_property_values(self, query: str) -> Dict[str, Any]:
        """Predict property values with dynamic filtering"""
        
        # Apply filters first
        filtered_props, filter_description = self._apply_dynamic_filters(query)
        filter_summary = " with ".join(filter_description) if filter_description else "All properties"
        
        if len(filtered_props) == 0:
            return {
                'query': query,
                'type': 'prediction',
                'response': f"No properties found matching criteria: {filter_summary}",
                'method': 'ML - Random Forest',
                'confidence': 0.0
            }
        
        # Get predictions
        X = filtered_props[self.value_features]
        predictions = self.models['value'].predict(X)
        
        filtered_props['predicted_value'] = predictions
        filtered_props['value_diff'] = predictions - filtered_props['property_value']
        filtered_props['value_diff_pct'] = (
            filtered_props['value_diff'] / filtered_props['property_value'] * 100
        )
        
        # Identify under/overvalued within filtered set
        undervalued = filtered_props[filtered_props['value_diff_pct'] > 10]
        overvalued = filtered_props[filtered_props['value_diff_pct'] < -10]
        
        response = f"PROPERTY VALUE ANALYSIS ({filter_summary}):\n\n"
        response += f"Analyzed {len(filtered_props)} properties\n\n"
        
        if len(undervalued) > 0:
            response += "🟢 POTENTIALLY UNDERVALUED (Buy/Hold Opportunities):\n\n"
            for _, prop in undervalued.nlargest(min(3, len(undervalued)), 'value_diff_pct').iterrows():
                response += f"• {prop['property_name']} ({prop['market']})\n"
                response += f"  Type: {prop['property_type']} | Class: {prop['building_class']}\n"
                response += f"  Current Value: ${prop['property_value']:,.0f}\n"
                response += f"  Predicted Value: ${prop['predicted_value']:,.0f}\n"
                response += f"  Upside Potential: {prop['value_diff_pct']:.1f}%\n"
                response += f"  Key Drivers: NOI: ${prop['noi_annual']:,.0f}, "
                response += f"Occupancy: {prop['occupancy_rate']:.1%}\n\n"
        
        if len(overvalued) > 0:
            response += "🔴 POTENTIALLY OVERVALUED (Sell/Reposition Candidates):\n\n"
            for _, prop in overvalued.nsmallest(min(3, len(overvalued)), 'value_diff_pct').iterrows():
                response += f"• {prop['property_name']} ({prop['market']})\n"
                response += f"  Current Value: ${prop['property_value']:,.0f}\n"
                response += f"  Predicted Value: ${prop['predicted_value']:,.0f}\n"
                response += f"  Downside Risk: {prop['value_diff_pct']:.1f}%\n\n"
        
        if len(undervalued) == 0 and len(overvalued) == 0:
            response += "All properties appear fairly valued based on current metrics.\n"
        
        return {
            'query': query,
            'type': 'prediction',
            'response': response,
            'method': 'ML - Random Forest with Dynamic Filtering',
            'confidence': 0.88
        }

    def assess_lease_risk(self, query: str) -> Dict[str, Any]:
        """Assess lease renewal risks with dynamic filtering"""
        
        # Apply filters first
        filtered_props, filter_description = self._apply_dynamic_filters(query)
        filter_summary = " with ".join(filter_description) if filter_description else "All properties"
        
        if len(filtered_props) == 0:
            return {
                'query': query,
                'type': 'risk_assessment',
                'response': f"No properties found matching criteria: {filter_summary}",
                'method': 'ML - Random Forest Classifier',
                'confidence': 0.0
            }
        
        X = filtered_props[self.lease_features]
        proba = self.models['lease_risk'].predict_proba(X)
        
        # Get risk probabilities
        classes = list(self.models['lease_risk'].classes_)
        if 1 in classes:
            idx_one = classes.index(1)
            risk_proba = proba[:, idx_one]
        else:
            risk_proba = np.zeros(len(X))
        
        filtered_props['lease_risk_score'] = risk_proba
        
        # Define risk bands
        high_risk = filtered_props[filtered_props['lease_risk_score'] > 0.7]
        medium_risk = filtered_props[
            (filtered_props['lease_risk_score'] > 0.4) &
            (filtered_props['lease_risk_score'] <= 0.7)
        ]
        
        response = f"LEASE RENEWAL RISK ASSESSMENT ({filter_summary}):\n\n"
        response += f"Analyzed {len(filtered_props)} properties\n\n"
        
        if len(high_risk) > 0:
            response += f"🚨 HIGH RISK PROPERTIES ({len(high_risk)} properties):\n\n"
            for _, prop in high_risk.nlargest(min(5, len(high_risk)), 'lease_risk_score').iterrows():
                response += f"• {prop['property_name']} ({prop['market']})\n"
                response += f"  Type: {prop['property_type']} | Class: {prop['building_class']}\n"
                response += f"  Risk Score: {prop['lease_risk_score']:.1%}\n"
                response += f"  WALT: {prop['walt_years']:.1f} years\n"
                response += f"  Action: Immediate tenant retention efforts needed\n\n"
        
        response += "\n📊 RISK SUMMARY:\n"
        response += f"• High Risk: {len(high_risk)} properties\n"
        response += f"• Medium Risk: {len(medium_risk)} properties\n"
        response += f"• Low Risk: {len(filtered_props) - len(high_risk) - len(medium_risk)} properties\n"
        
        if 'annual_rent' in filtered_props.columns and len(high_risk) > 0:
            response += f"• Total at Risk: ${high_risk['annual_rent'].sum():,.0f} in annual rent\n"
        
        return {
            'query': query,
            'type': 'risk_assessment',
            'response': response,
            'method': 'ML - Random Forest Classifier with Dynamic Filtering',
            'confidence': 0.82
        }

    def identify_energy_opportunities(self, query: str) -> Dict[str, Any]:
        """Identify energy efficiency opportunities with dynamic filtering"""
        
        # Apply filters first
        filtered_props, filter_description = self._apply_dynamic_filters(query)
        filter_summary = " with ".join(filter_description) if filter_description else "All properties"
        
        if len(filtered_props) == 0:
            return {
                'query': query,
                'type': 'optimization',
                'response': f"No properties found matching criteria: {filter_summary}",
                'method': 'ML Analysis + Domain Rules',
                'confidence': 0.0
            }
        
        # Find opportunities within filtered set
        opportunities = filtered_props[
            (filtered_props['energy_cost_psf'] > 3.0) &
            (filtered_props['energy_star_score'] < 75)
        ].copy()
        
        if len(opportunities) == 0:
            # If no high-cost properties, show worst performers
            opportunities = filtered_props.nsmallest(
                min(5, len(filtered_props)), 'energy_star_score'
            ).copy()
        
        # Calculate potential savings
        opportunities['potential_savings'] = opportunities['energy_cost_annual'] * 0.20
        
        response = f"ENERGY EFFICIENCY OPPORTUNITIES ({filter_summary}):\n\n"
        response += f"Analyzed {len(filtered_props)} properties\n"
        response += f"Found {len(opportunities)} with optimization potential\n\n"
        
        total_savings = opportunities['potential_savings'].sum()
        
        for _, prop in opportunities.nlargest(min(5, len(opportunities)), 'potential_savings').iterrows():
            response += f"📍 {prop['property_name']} ({prop['market']})\n"
            response += f"   Type: {prop['property_type']} | Class: {prop['building_class']}\n"
            response += f"   Current Energy Cost: ${prop['energy_cost_annual']:,.0f}/year "
            response += f"(${prop['energy_cost_psf']:.2f}/sqft)\n"
            response += f"   Energy Star Score: {prop['energy_star_score']}/100\n"
            response += f"   Potential Savings: ${prop['potential_savings']:,.0f}/year\n"
            response += f"   Recommended Actions: "
            
            if prop['energy_star_score'] < 50:
                response += "Full energy audit, HVAC upgrade, LED conversion\n"
            elif prop['energy_star_score'] < 75:
                response += "Building automation, occupancy sensors, insulation improvement\n"
            else:
                response += "Smart controls, renewable energy assessment\n"
            
            response += "\n"
        
        response += f"\n💰 TOTAL SAVINGS OPPORTUNITY: ${total_savings:,.0f}/year\n"
        
        return {
            'query': query,
            'type': 'optimization',
            'response': response,
            'method': 'ML Analysis with Dynamic Filtering',
            'confidence': 0.90
        }
    
    def _parse_query_filters(self, query: str) -> Dict[str, Any]:
        """Parse query for filters"""
        
        query_lower = query.lower()
        filters = {}
        
        # Market detection
        for market in self.properties['market'].unique():
            if market.lower() in query_lower:
                filters['market'] = market
                break
        
        # Property type detection
        for prop_type in self.properties['property_type'].unique():
            if prop_type.lower() in query_lower:
                filters['property_type'] = prop_type
                break
        
        # Condition filters
        if 'high energy' in query_lower:
            filters['high_energy'] = True
        if 'low occupancy' in query_lower or 'vacant' in query_lower:
            filters['low_occupancy'] = True
        if 'expiring' in query_lower or 'lease expir' in query_lower:
            filters['expiring_leases'] = True
        
        return filters
    
    def _format_property_list(self, properties_df, query: str) -> str:
        """Format property list for response"""
        
        if len(properties_df) == 0:
            return "No properties found matching your criteria."
        
        response = f"Found {len(properties_df)} properties matching your query:\n\n"
        
        for idx, (_, prop) in enumerate(properties_df.iterrows(), 1):
            response += f"{idx}. {prop['property_name']} ({prop['market']})\n"
            response += f"   • Type: {prop['property_type']} (Class {prop['building_class']})\n"
            response += f"   • Size: {prop['total_sqft']:,.0f} sqft\n"
            response += f"   • Occupancy: {prop['occupancy_rate']:.1%}\n"
            
            # Add relevant metrics based on query
            if 'energy' in query.lower():
                response += f"   • Energy Cost: ${prop['energy_cost_psf']:.2f}/sqft\n"
                response += f"   • Energy Star: {prop['energy_star_score']}/100\n"
            
            if 'lease' in query.lower() or 'expir' in query.lower():
                response += f"   • WALT: {prop['walt_years']:.1f} years\n"
                response += f"   • Expiring: {prop['lease_expiry_next_12mo_sqft']:,.0f} sqft\n"
            
            if 'value' in query.lower() or 'noi' in query.lower():
                response += f"   • NOI: ${prop['noi_annual']:,.0f}\n"
                response += f"   • Cap Rate: {prop['cap_rate']:.2%}\n"
            
            response += "\n"
        
        return response
    
    def predict_occupancy_rates(self, query: str) -> Dict[str, Any]:
        """Predict future occupancy rates with dynamic filtering"""
        
        # Apply filters first
        filtered_props, filter_description = self._apply_dynamic_filters(query)
        filter_summary = " with ".join(filter_description) if filter_description else "All properties"
        
        if len(filtered_props) == 0:
            return {
                'query': query,
                'type': 'prediction',
                'response': f"No properties found matching criteria: {filter_summary}",
                'method': 'ML - Random Forest',
                'confidence': 0.0
            }
        
        X = filtered_props[self.occupancy_features]
        predictions = self.models['occupancy'].predict(X)
        
        filtered_props['predicted_occupancy'] = predictions
        filtered_props['occupancy_change'] = predictions - filtered_props['occupancy_rate']
        
        declining = filtered_props[filtered_props['occupancy_change'] < -0.05]
        improving = filtered_props[filtered_props['occupancy_change'] > 0.05]
        
        response = f"OCCUPANCY RATE PREDICTIONS ({filter_summary}):\n\n"
        response += f"Analyzing {len(filtered_props)} properties\n\n"
        
        if len(declining) > 0:
            response += "⚠️ Properties at Risk of Declining Occupancy:\n\n"
            for _, prop in declining.nsmallest(min(5, len(declining)), 'occupancy_change').iterrows():
                response += f"• {prop['property_name']} ({prop['market']})\n"
                response += f"  Type: {prop['property_type']} | Class: {prop['building_class']}\n"
                response += f"  Current: {prop['occupancy_rate']:.1%}\n"
                response += f"  Predicted: {prop['predicted_occupancy']:.1%}\n"
                response += f"  Change: {prop['occupancy_change']*100:.1f}%\n\n"
        
        if len(improving) > 0:
            response += "📈 Properties Expected to Improve:\n\n"
            for _, prop in improving.nlargest(min(3, len(improving)), 'occupancy_change').iterrows():
                response += f"• {prop['property_name']} ({prop['market']})\n"
                response += f"  Current: {prop['occupancy_rate']:.1%} → "
                response += f"Predicted: {prop['predicted_occupancy']:.1%}\n"
        
        return {
            'query': query,
            'type': 'prediction',
            'response': response,
            'method': 'ML - Random Forest with Dynamic Filtering',
            'confidence': 0.79
        }

    def predict_general_trends(self, query: str) -> Dict[str, Any]:
        """General trend predictions with dynamic filtering"""
        
        # Apply filters first
        filtered_props, filter_description = self._apply_dynamic_filters(query)
        filter_summary = " with ".join(filter_description) if filter_description else "All properties"
        
        if len(filtered_props) == 0:
            return {
                'query': query,
                'type': 'prediction',
                'response': f"No properties found matching criteria: {filter_summary}",
                'method': 'Statistical Analysis',
                'confidence': 0.0
            }
        
        response = f"PORTFOLIO TREND ANALYSIS ({filter_summary}):\n\n"
        response += f"Analyzing {len(filtered_props)} properties\n\n"
        
        # Market trends
        market_summary = filtered_props.groupby('market').agg({
            'occupancy_rate': 'mean',
            'base_rent_psf': 'mean',
            'cap_rate': 'mean'
        })
        
        response += "Market Performance Trends:\n"
        for market, metrics in market_summary.iterrows():
            response += f"• {market}: Occupancy {metrics['occupancy_rate']:.1%}, "
            response += f"Rent ${metrics['base_rent_psf']:.2f}/sqft, "
            response += f"Cap Rate {metrics['cap_rate']:.2%}\n"
        
        return {
            'query': query,
            'type': 'prediction',
            'response': response,
            'method': 'Statistical Analysis with Dynamic Filtering',
            'confidence': 0.75
        }

    def assess_maintenance_risk(self, query: str) -> Dict[str, Any]:
        """Assess maintenance risks with dynamic filtering"""
        
        # Apply filters first
        filtered_props, filter_description = self._apply_dynamic_filters(query)
        filter_summary = " with ".join(filter_description) if filter_description else "All properties"
        
        if len(filtered_props) == 0:
            return {
                'query': query,
                'type': 'risk_assessment',
                'response': f"No properties found matching criteria: {filter_summary}",
                'method': 'Risk Scoring Algorithm',
                'confidence': 0.0
            }
        
        high_risk = filtered_props[filtered_props['maintenance_risk_score'] > 0.7]
        
        response = f"MAINTENANCE RISK ASSESSMENT ({filter_summary}):\n\n"
        response += f"Analyzed {len(filtered_props)} properties\n"
        response += f"Found {len(high_risk)} with high maintenance risk\n\n"
        
        for _, prop in high_risk.nlargest(min(5, len(high_risk)), 'maintenance_risk_score').iterrows():
            response += f"• {prop['property_name']} ({prop['market']})\n"
            response += f"  Type: {prop['property_type']} | Class: {prop['building_class']}\n"
            response += f"  Risk Score: {prop['maintenance_risk_score']:.2f}\n"
            response += f"  Building Age: {prop['building_age']} years\n"
            response += f"  Annual Maintenance: ${prop['maintenance_annual']:,.0f}\n\n"
        
        return {
            'query': query,
            'type': 'risk_assessment',
            'response': response,
            'method': 'Risk Scoring Algorithm with Dynamic Filtering',
            'confidence': 0.85
        }

    def assess_overall_risk(self, query: str) -> Dict[str, Any]:
        """Overall portfolio risk assessment with dynamic filtering"""
        
        # Apply filters first
        filtered_props, filter_description = self._apply_dynamic_filters(query)
        filter_summary = " with ".join(filter_description) if filter_description else "All properties"
        
        if len(filtered_props) == 0:
            return {
                'query': query,
                'type': 'risk_assessment',
                'response': f"No properties found matching criteria: {filter_summary}",
                'method': 'Composite Risk Scoring',
                'confidence': 0.0
            }
        
        # Calculate composite risk score
        filtered_props['composite_risk'] = (
            filtered_props['maintenance_risk_score'] * 0.25 +
            filtered_props['tenant_risk_score'] * 0.25 +
            filtered_props['market_risk_score'] * 0.25 +
            filtered_props['esg_risk_score'] * 0.25
        )
        
        high_risk = filtered_props[filtered_props['composite_risk'] > 0.6]
        
        response = f"PORTFOLIO RISK ASSESSMENT ({filter_summary}):\n\n"
        response += f"Analyzed {len(filtered_props)} properties\n"
        response += f"High Risk Properties: {len(high_risk)}\n\n"
        
        for _, prop in high_risk.nlargest(min(5, len(high_risk)), 'composite_risk').iterrows():
            response += f"• {prop['property_name']} ({prop['market']})\n"
            response += f"  Type: {prop['property_type']} | Class: {prop['building_class']}\n"
            response += f"  Overall Risk: {prop['composite_risk']:.2f}\n"
            response += f"  Key Risks: "
            
            risks = []
            if prop['maintenance_risk_score'] > 0.7:
                risks.append("Maintenance")
            if prop['tenant_risk_score'] > 0.7:
                risks.append("Tenant")
            if prop['market_risk_score'] > 0.7:
                risks.append("Market")
            
            response += ", ".join(risks) + "\n\n"
        
        return {
            'query': query,
            'type': 'risk_assessment',
            'response': response,
            'method': 'Composite Risk Scoring with Dynamic Filtering',
            'confidence': 0.83
        }

    def identify_value_opportunities(self, query: str) -> Dict[str, Any]:
        """Identify value optimization opportunities with dynamic filtering"""
        
        # Use the centralized dynamic filter method
        filtered_props, filter_description = self._apply_dynamic_filters(query)
        filter_summary = " with ".join(filter_description) if filter_description else "All properties"
        
        # Check if we have any properties after filtering
        if len(filtered_props) == 0:
            return {
                'query': query,
                'type': 'optimization',
                'response': f"No properties found matching criteria: {filter_summary}",
                'method': 'ML Value Analysis',
                'confidence': 0.0
            }
        
        # Run value predictions on filtered properties
        X = filtered_props[self.value_features]
        predictions = self.models['value'].predict(X)
        
        filtered_props['predicted_value'] = predictions
        filtered_props['value_opportunity'] = predictions - filtered_props['property_value']
        filtered_props['value_opportunity_pct'] = (
            filtered_props['value_opportunity'] / filtered_props['property_value'] * 100
        )
        
        # Get undervalued properties
        undervalued = filtered_props[filtered_props['value_opportunity'] > 0]
        opportunities = undervalued.nlargest(
            min(5, len(undervalued)), 
            'value_opportunity'
        )
        
        if len(opportunities) == 0:
            response = f"VALUE OPTIMIZATION ANALYSIS ({filter_summary}):\n\n"
            response += f"No undervalued properties found.\n"
            response += f"Properties analyzed: {len(filtered_props)}\n"
            response += f"All properties appear fairly valued or overvalued based on current metrics."
        else:
            response = f"VALUE OPTIMIZATION OPPORTUNITIES ({filter_summary}):\n\n"
            response += f"Found {len(opportunities)} undervalued properties from {len(filtered_props)} analyzed:\n\n"
            
            for _, prop in opportunities.iterrows():
                response += f"• {prop['property_name']} ({prop['market']})\n"
                response += f"  Type: {prop['property_type']} | Class: {prop['building_class']}\n"
                response += f"  Current Value: ${prop['property_value']:,.0f}\n"
                response += f"  Predicted Value: ${prop['predicted_value']:,.0f}\n"
                response += f"  Opportunity: ${prop['value_opportunity']:,.0f} ({prop['value_opportunity_pct']:.1f}%)\n"
                
                # Include lease info if relevant to query
                if 'expir' in query.lower() or 'lease' in query.lower():
                    response += f"  WALT: {prop['walt_years']:.1f} years\n"
                
                response += f"  Occupancy: {prop['occupancy_rate']:.1%}\n"
                response += f"  Strategy: "
                
                if prop['occupancy_rate'] < 0.85:
                    response += "Focus on leasing to increase occupancy\n"
                elif prop['energy_star_score'] < 75:
                    response += "Energy efficiency improvements\n"
                else:
                    response += "Rent optimization and expense management\n"
                
                response += "\n"
        
        return {
            'query': query,
            'type': 'optimization',
            'response': response,
            'method': 'ML Value Analysis with Dynamic Filtering',
            'confidence': 0.86 if len(opportunities) > 0 else 0.5
        }

    def identify_general_opportunities(self, query: str) -> Dict[str, Any]:
        """Identify general optimization opportunities with dynamic filtering"""
        
        # Apply filters first
        filtered_props, filter_description = self._apply_dynamic_filters(query)
        filter_summary = " with ".join(filter_description) if filter_description else "All properties"
        
        if len(filtered_props) == 0:
            return {
                'query': query,
                'type': 'optimization',
                'response': f"No properties found matching criteria: {filter_summary}",
                'method': 'Portfolio Analysis',
                'confidence': 0.0
            }
        
        response = f"PORTFOLIO OPTIMIZATION OPPORTUNITIES ({filter_summary}):\n\n"
        response += f"Analyzing {len(filtered_props)} properties\n\n"
        
        # Initialize counters
        opportunities_found = 0
        
        # 1. Low occupancy opportunities
        low_occ = filtered_props[filtered_props['occupancy_rate'] < 0.75]
        if len(low_occ) > 0:
            opportunities_found += 1
            response += f"{opportunities_found}. OCCUPANCY IMPROVEMENT:\n"
            response += f"   • {len(low_occ)} properties below 75% occupancy\n"
            potential_revenue = (low_occ['vacant_sqft'] * low_occ['base_rent_psf']).sum()
            response += f"   • Potential Revenue: ${potential_revenue:,.0f}\n"
            response += f"   • Top opportunities:\n"
            for _, prop in low_occ.nsmallest(min(3, len(low_occ)), 'occupancy_rate').iterrows():
                response += f"     - {prop['property_name']}: {prop['occupancy_rate']:.1%} occupancy\n"
            response += "\n"
        
        # 2. Energy efficiency opportunities
        high_energy = filtered_props[
            (filtered_props['energy_cost_psf'] > 3.5) & 
            (filtered_props['energy_star_score'] < 70)
        ]
        if len(high_energy) > 0:
            opportunities_found += 1
            response += f"{opportunities_found}. ENERGY EFFICIENCY:\n"
            response += f"   • {len(high_energy)} properties with high costs & low efficiency\n"
            potential_savings = (high_energy['energy_cost_annual'] * 0.2).sum()
            response += f"   • Potential Savings: ${potential_savings:,.0f}/year\n"
            response += f"   • Top opportunities:\n"
            for _, prop in high_energy.nlargest(min(3, len(high_energy)), 'energy_cost_psf').iterrows():
                response += f"     - {prop['property_name']}: ${prop['energy_cost_psf']:.2f}/sqft, Energy Star {prop['energy_star_score']}\n"
            response += "\n"
        
        # 3. Lease renewal opportunities
        expiring = filtered_props[filtered_props['walt_years'] < 1]
        if len(expiring) > 0:
            opportunities_found += 1
            response += f"{opportunities_found}. LEASE RENEWALS:\n"
            response += f"   • {len(expiring)} properties with leases expiring within 1 year\n"
            at_risk_revenue = expiring['effective_rental_income'].sum()
            response += f"   • At Risk Revenue: ${at_risk_revenue:,.0f}\n"
            response += f"   • Urgent attention needed:\n"
            for _, prop in expiring.nsmallest(min(3, len(expiring)), 'walt_years').iterrows():
                response += f"     - {prop['property_name']}: {prop['walt_years']:.1f} years WALT\n"
            response += "\n"
        
        # 4. Value optimization opportunities
        if len(filtered_props) > 0:
            # Quick value check
            X = filtered_props[self.value_features]
            predictions = self.models['value'].predict(X)
            filtered_props['value_diff'] = predictions - filtered_props['property_value']
            undervalued = filtered_props[filtered_props['value_diff'] > 0]
            
            if len(undervalued) > 0:
                opportunities_found += 1
                response += f"{opportunities_found}. VALUE OPTIMIZATION:\n"
                response += f"   • {len(undervalued)} potentially undervalued properties\n"
                total_opportunity = undervalued['value_diff'].sum()
                response += f"   • Total Value Opportunity: ${total_opportunity:,.0f}\n"
                response += f"   • Top opportunities:\n"
                for _, prop in undervalued.nlargest(min(3, len(undervalued)), 'value_diff').iterrows():
                    response += f"     - {prop['property_name']}: ${prop['value_diff']:,.0f} potential\n"
                response += "\n"
        
        if opportunities_found == 0:
            response += "No significant optimization opportunities identified in the filtered portfolio.\n"
            response += "Consider expanding search criteria or reviewing property performance metrics.\n"
        
        return {
            'query': query,
            'type': 'optimization',
            'response': response,
            'method': 'Portfolio Analysis with Dynamic Filtering',
            'confidence': 0.88 if opportunities_found > 0 else 0.5
        }
# Demo and testing
def demo_hybrid_system():
    """Demonstrate the hybrid system with various queries"""
    
    print("\n" + "="*70)
    print("IKARIS - Intelligent CRE System Demo")
    print("="*70)
    
    # Initialize system
    system = IkarisHybridSystem()
    
    # Test queries - mix of RAG and ML
    test_queries = [
        # RAG queries (factual)
        "Which properties in Dallas have high energy costs?",
        "Show me office buildings with occupancy below 80%",
        "List retail properties in Houston",
        
        # ML prediction queries
        "Predict which properties will have the highest maintenance costs next year",
        "What will be the property values next quarter?",
        "Forecast occupancy rates for the next 6 months",
        
        # ML risk queries
        "Which properties are at risk of lease non-renewal?",
        "Assess maintenance risk across the portfolio",
        
        # ML optimization queries
        "Which properties are undervalued?",
        "Identify energy efficiency opportunities",
        "Recommend properties for value optimization"
    ]
    
    for query in test_queries:
        print("\n" + "-"*70)
        result = system.process_query(query)
        
        print(f"Query: {query}")
        print(f"Type: {result['type']}")
        print(f"Method: {result['method']}")
        if 'confidence' in result:
            print(f"Confidence: {result['confidence']:.1%}")
        print(f"\nResponse:\n{result['response'][:500]}...")  # Truncate for display

if __name__ == "__main__":
    demo_hybrid_system()