"""
CBRE Commercial Real Estate Data Generator
Generates realistic synthetic data for HackUTD 2025 Challenge
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
from typing import Dict, List, Tuple

class CBREDataGenerator:
    """Generate realistic commercial real estate data for CBRE hackathon"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Real market data based on Q3 2024 CBRE reports
        self.markets = {
            'Dallas': {
                'market_tier': 1,
                'office_vacancy': 0.192,
                'office_rent_psf': 28.50,
                'industrial_vacancy': 0.074,
                'industrial_rent_psf': 8.25,
                'retail_vacancy': 0.054,
                'retail_rent_psf': 22.00,
                'cap_rate_office': 0.072,
                'cap_rate_industrial': 0.058,
                'cap_rate_retail': 0.068,
                'market_growth': 0.035
            },
            'Austin': {
                'market_tier': 1,
                'office_vacancy': 0.218,
                'office_rent_psf': 42.75,
                'industrial_vacancy': 0.081,
                'industrial_rent_psf': 11.50,
                'retail_vacancy': 0.048,
                'retail_rent_psf': 28.00,
                'cap_rate_office': 0.068,
                'cap_rate_industrial': 0.055,
                'cap_rate_retail': 0.065,
                'market_growth': 0.042
            },
            'Houston': {
                'market_tier': 1,
                'office_vacancy': 0.224,
                'office_rent_psf': 26.25,
                'industrial_vacancy': 0.069,
                'industrial_rent_psf': 7.75,
                'retail_vacancy': 0.062,
                'retail_rent_psf': 19.50,
                'cap_rate_office': 0.075,
                'cap_rate_industrial': 0.060,
                'cap_rate_retail': 0.070,
                'market_growth': 0.028
            },
            'San Antonio': {
                'market_tier': 2,
                'office_vacancy': 0.156,
                'office_rent_psf': 24.00,
                'industrial_vacancy': 0.065,
                'industrial_rent_psf': 7.00,
                'retail_vacancy': 0.058,
                'retail_rent_psf': 18.75,
                'cap_rate_office': 0.078,
                'cap_rate_industrial': 0.062,
                'cap_rate_retail': 0.072,
                'market_growth': 0.031
            },
            'New York': {
                'market_tier': 1,
                'office_vacancy': 0.168,
                'office_rent_psf': 72.50,
                'industrial_vacancy': 0.042,
                'industrial_rent_psf': 18.00,
                'retail_vacancy': 0.112,
                'retail_rent_psf': 85.00,
                'cap_rate_office': 0.055,
                'cap_rate_industrial': 0.048,
                'cap_rate_retail': 0.052,
                'market_growth': 0.022
            },
            'Los Angeles': {
                'market_tier': 1,
                'office_vacancy': 0.195,
                'office_rent_psf': 45.00,
                'industrial_vacancy': 0.038,
                'industrial_rent_psf': 14.50,
                'retail_vacancy': 0.068,
                'retail_rent_psf': 38.00,
                'cap_rate_office': 0.062,
                'cap_rate_industrial': 0.050,
                'cap_rate_retail': 0.058,
                'market_growth': 0.025
            },
            'Chicago': {
                'market_tier': 1,
                'office_vacancy': 0.201,
                'office_rent_psf': 32.00,
                'industrial_vacancy': 0.058,
                'industrial_rent_psf': 6.50,
                'retail_vacancy': 0.075,
                'retail_rent_psf': 24.00,
                'cap_rate_office': 0.070,
                'cap_rate_industrial': 0.056,
                'cap_rate_retail': 0.066,
                'market_growth': 0.018
            },
            'Miami': {
                'market_tier': 1,
                'office_vacancy': 0.142,
                'office_rent_psf': 48.50,
                'industrial_vacancy': 0.045,
                'industrial_rent_psf': 12.00,
                'retail_vacancy': 0.052,
                'retail_rent_psf': 42.00,
                'cap_rate_office': 0.065,
                'cap_rate_industrial': 0.052,
                'cap_rate_retail': 0.060,
                'market_growth': 0.038
            }
        }
        
        # Property types with realistic characteristics
        self.property_types = {
            'Office': {
                'avg_size': 75000,
                'size_std': 25000,
                'floors_range': (3, 40),
                'typical_tenants': 15,
                'energy_intensity': 55,  # kBtu/sqft/year
                'maintenance_factor': 1.0
            },
            'Industrial': {
                'avg_size': 120000,
                'size_std': 40000,
                'floors_range': (1, 3),
                'typical_tenants': 3,
                'energy_intensity': 35,
                'maintenance_factor': 0.7
            },
            'Retail': {
                'avg_size': 45000,
                'size_std': 20000,
                'floors_range': (1, 4),
                'typical_tenants': 8,
                'energy_intensity': 48,
                'maintenance_factor': 0.9
            },
            'Multifamily': {
                'avg_size': 95000,
                'size_std': 30000,
                'floors_range': (3, 25),
                'typical_tenants': 100,  # units
                'energy_intensity': 42,
                'maintenance_factor': 0.85
            },
            'Mixed Use': {
                'avg_size': 110000,
                'size_std': 35000,
                'floors_range': (5, 30),
                'typical_tenants': 25,
                'energy_intensity': 50,
                'maintenance_factor': 1.1
            },
            'Data Center': {
                'avg_size': 50000,
                'size_std': 20000,
                'floors_range': (1, 4),
                'typical_tenants': 5,
                'energy_intensity': 150,  # Very high energy use
                'maintenance_factor': 1.5
            },
            'Medical Office': {
                'avg_size': 40000,
                'size_std': 15000,
                'floors_range': (2, 8),
                'typical_tenants': 12,
                'energy_intensity': 65,
                'maintenance_factor': 1.2
            }
        }
        
        # Building classes
        self.building_classes = {
            'A': {'quality_factor': 1.3, 'prob': 0.25},
            'B': {'quality_factor': 1.0, 'prob': 0.50},
            'C': {'quality_factor': 0.75, 'prob': 0.25}
        }
        
        # Tenant credit ratings distribution
        self.credit_ratings = {
            'AAA': 0.05, 'AA': 0.10, 'A': 0.20,
            'BBB': 0.30, 'BB': 0.20, 'B': 0.10, 'CCC': 0.05
        }

    def generate_properties(self, n_properties: int = 500) -> pd.DataFrame:
        """Generate main property dataset"""
        
        properties = []
        
        for i in range(n_properties):
            # Select random characteristics
            market = random.choice(list(self.markets.keys()))
            prop_type = np.random.choice(
                list(self.property_types.keys()),
                p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]  # Office heavy portfolio
            )
            
            market_data = self.markets[market]
            type_data = self.property_types[prop_type]
            
            # Building characteristics
            building_class = np.random.choice(
                list(self.building_classes.keys()),
                p=[self.building_classes[c]['prob'] for c in self.building_classes]
            )
            
            # Size based on property type
            size = max(10000, np.random.normal(type_data['avg_size'], type_data['size_std']))
            floors = np.random.randint(*type_data['floors_range'])
            
            # Age and condition
            year_built = np.random.choice(
                range(1970, 2025),
                p=np.linspace(0.01, 0.03, 55) / np.linspace(0.01, 0.03, 55).sum()
            )
            building_age = 2024 - year_built
            
            # Financial metrics
            if prop_type == 'Office':
                base_rent = market_data['office_rent_psf']
                vacancy = market_data['office_vacancy']
                cap_rate = market_data['cap_rate_office']
            elif prop_type == 'Industrial':
                base_rent = market_data['industrial_rent_psf']
                vacancy = market_data['industrial_vacancy']
                cap_rate = market_data['cap_rate_industrial']
            elif prop_type == 'Retail':
                base_rent = market_data['retail_rent_psf']
                vacancy = market_data['retail_vacancy']
                cap_rate = market_data['cap_rate_retail']
            else:  # Mixed use, multifamily, etc
                base_rent = market_data['office_rent_psf'] * 0.8
                vacancy = market_data['office_vacancy'] * 0.7
                cap_rate = market_data['cap_rate_office'] * 0.95
            
            # Apply building class adjustments
            quality_factor = self.building_classes[building_class]['quality_factor']
            base_rent *= quality_factor
            
            # Add some randomness
            base_rent *= np.random.uniform(0.85, 1.15)
            
            # Occupancy (inverse of vacancy with randomness)
            occupancy = max(0.5, min(0.99, 1 - vacancy * np.random.uniform(0.7, 1.3)))
            
            # Operating expenses (realistic ranges per sqft)
            opex = {
                'property_tax': np.random.uniform(2.0, 5.0),
                'insurance': np.random.uniform(0.30, 0.80),
                'utilities': np.random.uniform(1.50, 3.50),
                'maintenance': np.random.uniform(1.00, 2.50) * type_data['maintenance_factor'],
                'management': base_rent * occupancy * 0.04,
                'cam': np.random.uniform(2.00, 4.00)
            }
            total_opex = sum(opex.values())
            
            # Calculate NOI and value
            potential_rental_income = size * base_rent
            effective_rental_income = potential_rental_income * occupancy * 0.98  # 2% credit loss
            noi = effective_rental_income - (size * total_opex)
            
            # Property value based on cap rate
            property_value = noi / cap_rate if cap_rate > 0 else 0
            
            # Lease information
            walt = self._calculate_walt(prop_type, occupancy)
            num_tenants = self._calculate_tenant_count(prop_type, size)
            
            # Sustainability metrics
            leed_certified = (year_built > 2010 and random.random() < 0.4) or \
                           (year_built > 2000 and random.random() < 0.2)
            
            energy_star_score = self._calculate_energy_star_score(
                year_built, leed_certified, building_class
            )
            
            # Calculate waste diversion rate based on LEED status
            waste_diversion_rate = round(np.random.uniform(0.25, 0.85) if leed_certified else np.random.uniform(0.10, 0.40), 2)
            
            # Create property record
            property_data = {
                'property_id': f'CBRE_{i+1:05d}',
                'property_name': f'{prop_type} at {market} - {i+1:03d}',
                'address': f'{np.random.randint(100, 9999)} {random.choice(["Main", "Market", "Commerce", "Business", "Park"])} {random.choice(["St", "Ave", "Blvd", "Way", "Plaza"])}',
                'city': market,
                'state': self._get_state(market),
                'zip_code': f'{np.random.randint(10000, 99999)}',
                'market': market,
                'submarket': f'{market}_Zone_{np.random.randint(1, 6)}',
                
                # Physical characteristics
                'property_type': prop_type,
                'building_class': building_class,
                'total_sqft': round(size),
                'rentable_sqft': round(size * 0.92),  # 92% efficiency
                'num_floors': floors,
                'year_built': year_built,
                'building_age': building_age,
                'last_renovated': year_built + np.random.randint(5, max(6, building_age)) if building_age > 10 else None,
                'parking_spaces': int(size / 1000) * np.random.randint(2, 5),
                'parking_ratio': round(int(size / 1000) * np.random.randint(2, 5) / (size / 1000), 2),
                
                # Financial metrics
                'property_value': round(property_value, 2),
                'value_per_sqft': round(property_value / size, 2),
                'noi_annual': round(noi, 2),
                'noi_per_sqft': round(noi / size, 2),
                'cap_rate': round(cap_rate, 4),
                'base_rent_psf': round(base_rent, 2),
                'effective_rent_psf': round(base_rent * 0.9, 2),  # After concessions
                'potential_rental_income': round(potential_rental_income, 2),
                'effective_rental_income': round(effective_rental_income, 2),
                
                # Occupancy metrics
                'occupancy_rate': round(occupancy, 3),
                'occupied_sqft': round(size * occupancy),
                'vacant_sqft': round(size * (1 - occupancy)),
                'market_vacancy_rate': round(vacancy, 3),
                
                # Operating expenses
                'opex_total_annual': round(size * total_opex, 2),
                'opex_per_sqft': round(total_opex, 2),
                'property_tax_annual': round(size * opex['property_tax'], 2),
                'insurance_annual': round(size * opex['insurance'], 2),
                'utilities_annual': round(size * opex['utilities'], 2),
                'maintenance_annual': round(size * opex['maintenance'], 2),
                'management_fee_annual': round(size * opex['management'], 2),
                'cam_annual': round(size * opex['cam'], 2),
                
                # Lease metrics
                'num_tenants': num_tenants,
                'walt_years': round(walt, 2),
                'walt_months': round(walt * 12, 1),
                'lease_expiry_next_12mo_sqft': round(size * min(0.3, 1/walt if walt > 0 else 0.3)),
                'avg_tenant_size_sqft': round(size * occupancy / num_tenants) if num_tenants > 0 else 0,
                'largest_tenant_sqft': round(size * occupancy * np.random.uniform(0.15, 0.40)) if num_tenants > 1 else round(size * occupancy),
                'tenant_concentration_risk': round(np.random.beta(2, 5), 3),  # Skewed low
                
                # Debt metrics
                'loan_amount': round(property_value * np.random.uniform(0.50, 0.75), 2),
                'loan_to_value': round(np.random.uniform(0.50, 0.75), 3),
                'debt_service_coverage_ratio': round(np.random.uniform(1.15, 2.00), 2),
                'debt_yield': round(np.random.uniform(0.08, 0.12), 3),
                
                # Sustainability & Energy
                'leed_certified': leed_certified,
                'leed_level': random.choice(['Certified', 'Silver', 'Gold', 'Platinum']) if leed_certified else None,
                'energy_star_score': energy_star_score,
                'energy_star_certified': energy_star_score >= 75,
                'annual_energy_use_kbtu': round(size * type_data['energy_intensity'] * np.random.uniform(0.8, 1.2)),
                'energy_cost_annual': round(size * np.random.uniform(2.0, 4.0)),
                'energy_cost_psf': round(np.random.uniform(2.0, 4.0), 2),
                'water_usage_gallons': round(size * np.random.uniform(10, 25)),
                'water_cost_annual': round(size * np.random.uniform(0.20, 0.80)),
                'waste_diversion_rate': waste_diversion_rate,
                'ghg_emissions_mtco2e': round(size * np.random.uniform(0.004, 0.008), 2),
                
                # Market metrics
                'market_tier': market_data['market_tier'],
                'market_rent_growth_yoy': round(market_data['market_growth'] * np.random.uniform(0.7, 1.3), 3),
                'market_cap_rate': round(cap_rate, 4),
                'market_absorption_sf': round(np.random.normal(50000, 20000)),
                
                # Risk scores (for ML predictions)
                'maintenance_risk_score': round(self._calculate_maintenance_risk(building_age, building_class), 2),
                'tenant_risk_score': round(self._calculate_tenant_risk(walt, occupancy, num_tenants), 2),
                'market_risk_score': round(self._calculate_market_risk(market_data, prop_type), 2),
                'esg_risk_score': round(self._calculate_esg_risk(energy_star_score, leed_certified, waste_diversion_rate), 2),
                
                # Additional metadata
                'management_company': random.choice(['CBRE', 'JLL', 'Cushman & Wakefield', 'Colliers', 'Newmark']),
                'acquisition_date': self._random_date(2015, 2023),
                'last_appraisal_date': self._random_date(2023, 2024),
                'last_inspection_date': self._random_date(2023, 2024)
            }
            
            properties.append(property_data)
        
        return pd.DataFrame(properties)
    
    def generate_tenant_data(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """Generate tenant details for properties"""
        
        tenant_data = []
        tenant_id = 1
        
        industries = ['Technology', 'Finance', 'Healthcare', 'Legal', 'Consulting', 
                     'Retail', 'Manufacturing', 'Government', 'Education', 'Non-Profit']
        
        for _, prop in properties_df.iterrows():
            num_tenants = prop['num_tenants']
            
            for t in range(int(num_tenants)):
                # Determine tenant size
                if t == 0:  # Largest tenant
                    tenant_sqft = prop['largest_tenant_sqft']
                else:
                    remaining_sqft = prop['occupied_sqft'] - prop['largest_tenant_sqft']
                    tenant_sqft = remaining_sqft / (num_tenants - 1) * np.random.uniform(0.5, 1.5)
                
                # Lease terms
                lease_start = self._random_date(2018, 2023)
                lease_term_years = np.random.choice([3, 5, 7, 10], p=[0.3, 0.4, 0.2, 0.1])
                lease_end = lease_start + timedelta(days=int(lease_term_years * 365))
                
                tenant = {
                    'tenant_id': f'T_{tenant_id:05d}',
                    'property_id': prop['property_id'],
                    'tenant_name': f'{random.choice(industries)} Corp {tenant_id}',
                    'industry': random.choice(industries),
                    'credit_rating': np.random.choice(
                        list(self.credit_ratings.keys()),
                        p=list(self.credit_ratings.values())
                    ),
                    'lease_sqft': round(tenant_sqft),
                    'lease_start_date': lease_start.strftime('%Y-%m-%d'),
                    'lease_end_date': lease_end.strftime('%Y-%m-%d'),
                    'lease_term_years': lease_term_years,
                    'base_rent_psf': prop['base_rent_psf'] * np.random.uniform(0.9, 1.1),
                    'annual_rent': round(tenant_sqft * prop['base_rent_psf'] * np.random.uniform(0.9, 1.1)),
                    'rent_escalation_rate': round(np.random.uniform(0.02, 0.04), 3),
                    'security_deposit': round(tenant_sqft * prop['base_rent_psf'] * 2),
                    'renewal_probability': round(np.random.uniform(0.6, 0.95), 2),
                    'payment_history_score': round(np.random.uniform(0.85, 1.0), 2),
                    'tenant_improvements_cost': round(tenant_sqft * np.random.uniform(20, 60))
                }
                
                tenant_data.append(tenant)
                tenant_id += 1
        
        return pd.DataFrame(tenant_data)
    
    def generate_historical_metrics(self, properties_df: pd.DataFrame, months: int = 24) -> pd.DataFrame:
        """Generate time series data for properties"""
        
        historical_data = []
        
        for _, prop in properties_df.iterrows():
            base_date = datetime.now() - timedelta(days=months * 30)
            
            for month in range(months):
                date = base_date + timedelta(days=month * 30)
                
                # Create trending metrics with seasonality
                seasonality = np.sin(2 * np.pi * month / 12) * 0.05
                trend = month / months * 0.02  # Slight upward trend
                
                historical = {
                    'property_id': prop['property_id'],
                    'date': date.strftime('%Y-%m-%d'),
                    'month': date.strftime('%Y-%m'),
                    'occupancy_rate': max(0.5, min(0.99, prop['occupancy_rate'] + seasonality + np.random.normal(0, 0.02))),
                    'effective_rent_psf': prop['effective_rent_psf'] * (1 + trend + np.random.normal(0, 0.01)),
                    'noi': prop['noi_annual'] / 12 * (1 + seasonality + trend + np.random.normal(0, 0.03)),
                    'energy_cost': prop['energy_cost_annual'] / 12 * (1 + seasonality * 2),  # More seasonal variation
                    'maintenance_cost': prop['maintenance_annual'] / 12 * (1 + np.random.normal(0, 0.1)),
                    'water_usage': prop['water_usage_gallons'] / 12 * (1 + seasonality),
                    'tenant_satisfaction_score': round(np.random.uniform(6.5, 9.0), 1),
                    'work_orders_count': np.random.poisson(5 + prop['building_age'] / 10),
                    'work_orders_avg_days': round(np.random.uniform(2, 10), 1),
                    'collections_rate': round(np.random.uniform(0.94, 0.99), 3)
                }
                
                historical_data.append(historical)
        
        return pd.DataFrame(historical_data)
    
    def generate_market_comparables(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """Generate market comparable properties for analysis"""
        
        comps = []
        
        for _, prop in properties_df.sample(min(100, len(properties_df))).iterrows():
            # Generate 3-5 comparables for each property
            num_comps = np.random.randint(3, 6)
            
            for c in range(num_comps):
                comp = {
                    'property_id': prop['property_id'],
                    'comp_property_id': f'COMP_{prop["property_id"]}_{c+1}',
                    'comp_address': f'{np.random.randint(100, 9999)} Comp Street',
                    'distance_miles': round(np.random.uniform(0.1, 5.0), 1),
                    'property_type': prop['property_type'],
                    'building_class': np.random.choice(['A', 'B', 'C'], p=[0.3, 0.5, 0.2]),
                    'year_built': prop['year_built'] + np.random.randint(-10, 10),
                    'total_sqft': prop['total_sqft'] * np.random.uniform(0.7, 1.3),
                    'occupancy_rate': prop['occupancy_rate'] * np.random.uniform(0.9, 1.1),
                    'asking_rent_psf': prop['base_rent_psf'] * np.random.uniform(0.85, 1.15),
                    'sale_price': prop['property_value'] * np.random.uniform(0.8, 1.2) if random.random() > 0.7 else None,
                    'sale_date': self._random_date(2023, 2024).strftime('%Y-%m-%d') if random.random() > 0.7 else None,
                    'cap_rate': prop['cap_rate'] * np.random.uniform(0.9, 1.1)
                }
                
                comps.append(comp)
        
        return pd.DataFrame(comps)
    
    def generate_document_references(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """Generate document metadata for RAG system integration"""
        
        documents = []
        doc_id = 1
        
        doc_types = {
            'Lease Agreement': {'count': 3, 'format': 'pdf'},
            'Property Inspection Report': {'count': 2, 'format': 'pdf'},
            'Financial Statement': {'count': 4, 'format': 'xlsx'},
            'Energy Audit': {'count': 1, 'format': 'pdf'},
            'Appraisal Report': {'count': 1, 'format': 'pdf'},
            'Insurance Policy': {'count': 1, 'format': 'pdf'},
            'Maintenance Log': {'count': 12, 'format': 'csv'},
            'Tenant Correspondence': {'count': 5, 'format': 'docx'},
            'Market Analysis': {'count': 1, 'format': 'pptx'},
            'Environmental Report': {'count': 1, 'format': 'pdf'}
        }
        
        for _, prop in properties_df.iterrows():
            for doc_type, info in doc_types.items():
                for i in range(np.random.randint(1, info['count'] + 1)):
                    doc = {
                        'document_id': f'DOC_{doc_id:06d}',
                        'property_id': prop['property_id'],
                        'document_type': doc_type,
                        'document_name': f'{prop["property_id"]}_{doc_type.replace(" ", "_")}_{i+1}.{info["format"]}',
                        'file_format': info['format'],
                        'file_size_mb': round(np.random.uniform(0.1, 5.0), 2),
                        'created_date': self._random_date(2020, 2024).strftime('%Y-%m-%d'),
                        'last_modified': self._random_date(2023, 2024).strftime('%Y-%m-%d'),
                        'document_status': np.random.choice(['Active', 'Draft', 'Archived'], p=[0.7, 0.2, 0.1]),
                        'access_level': np.random.choice(['Public', 'Internal', 'Confidential'], p=[0.2, 0.5, 0.3]),
                        'key_terms_extracted': self._generate_key_terms(doc_type),
                        'has_signatures': doc_type in ['Lease Agreement', 'Insurance Policy'],
                        'requires_renewal': doc_type in ['Insurance Policy', 'Lease Agreement'],
                        'expiry_date': (datetime.now() + timedelta(days=int(np.random.randint(30, 730)))).strftime('%Y-%m-%d') 
                                      if doc_type in ['Insurance Policy', 'Lease Agreement'] else None
                    }
                    
                    documents.append(doc)
                    doc_id += 1
        
        return pd.DataFrame(documents)
    
    # Helper methods
    def _calculate_walt(self, prop_type: str, occupancy: float) -> float:
        """Calculate Weighted Average Lease Term"""
        base_walt = {'Office': 5, 'Industrial': 3, 'Retail': 7, 
                    'Multifamily': 1, 'Mixed Use': 4, 'Data Center': 10,
                    'Medical Office': 7}
        
        walt = base_walt.get(prop_type, 5) * occupancy * np.random.uniform(0.3, 1.2)
        return max(0.5, min(10, walt))
    
    def _calculate_tenant_count(self, prop_type: str, size: float) -> int:
        """Calculate number of tenants based on property type and size"""
        if prop_type == 'Multifamily':
            return int(size / 1000)  # Units
        elif prop_type == 'Industrial':
            return max(1, np.random.poisson(2))
        else:
            return max(1, int(np.random.lognormal(np.log(size / 10000), 0.5)))
    
    def _calculate_energy_star_score(self, year_built: int, leed: bool, building_class: str) -> int:
        """Calculate Energy Star score"""
        base_score = 50
        
        # Age factor
        age = 2024 - year_built
        if age < 5:
            base_score += 20
        elif age < 15:
            base_score += 10
        elif age > 30:
            base_score -= 10
        
        # LEED bonus
        if leed:
            base_score += 25
        
        # Building class factor
        class_bonus = {'A': 15, 'B': 5, 'C': -5}
        base_score += class_bonus[building_class]
        
        # Add randomness
        score = base_score + np.random.randint(-10, 10)
        
        return max(1, min(100, score))
    
    def _calculate_maintenance_risk(self, age: int, building_class: str) -> float:
        """Calculate maintenance risk score (0-1)"""
        age_risk = min(1.0, age / 50)
        class_risk = {'A': 0.2, 'B': 0.5, 'C': 0.8}[building_class]
        return (age_risk + class_risk) / 2 * np.random.uniform(0.8, 1.2)
    
    def _calculate_tenant_risk(self, walt: float, occupancy: float, num_tenants: int) -> float:
        """Calculate tenant risk score (0-1)"""
        walt_risk = max(0, 1 - walt / 5)  # Higher risk if WALT < 5 years
        occupancy_risk = 1 - occupancy
        concentration_risk = 1 / max(1, num_tenants)  # Higher risk with fewer tenants
        
        return (walt_risk + occupancy_risk + concentration_risk) / 3
    
    def _calculate_market_risk(self, market_data: Dict, prop_type: str) -> float:
        """Calculate market risk score (0-1)"""
        vacancy_risk = market_data.get(f'{prop_type.lower()}_vacancy', 0.15)
        growth_risk = max(0, 1 - market_data['market_growth'] * 10)
        tier_risk = (3 - market_data['market_tier']) / 3
        
        return (vacancy_risk + growth_risk + tier_risk) / 3
    
    def _calculate_esg_risk(self, energy_star: int, leed: bool, waste_rate: float) -> float:
        """Calculate ESG risk score (0-1)"""
        energy_risk = (100 - energy_star) / 100
        cert_risk = 0 if leed else 0.5
        waste_risk = 1 - waste_rate
        
        return (energy_risk + cert_risk + waste_risk) / 3
    
    def _get_state(self, city: str) -> str:
        """Get state from city"""
        city_state = {
            'Dallas': 'TX', 'Austin': 'TX', 'Houston': 'TX', 'San Antonio': 'TX',
            'New York': 'NY', 'Los Angeles': 'CA', 'Chicago': 'IL', 'Miami': 'FL'
        }
        return city_state.get(city, 'TX')
    
    def _random_date(self, start_year: int, end_year: int):
        """Generate random date between years"""
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        days_between = (end_date - start_date).days
        random_days = int(np.random.randint(0, days_between))
        return start_date + timedelta(days=random_days)
    
    def _generate_key_terms(self, doc_type: str) -> str:
        """Generate key terms for document"""
        terms = {
            'Lease Agreement': ['rent', 'term', 'renewal', 'escalation', 'cam'],
            'Property Inspection Report': ['hvac', 'roof', 'structural', 'safety', 'code'],
            'Financial Statement': ['noi', 'revenue', 'expenses', 'capex', 'cash flow'],
            'Energy Audit': ['consumption', 'efficiency', 'hvac', 'lighting', 'recommendations'],
            'Appraisal Report': ['value', 'comparables', 'market', 'highest best use', 'cap rate']
        }
        
        return ', '.join(terms.get(doc_type, ['general', 'property', 'management']))
    
    def generate_all_datasets(self, n_properties: int = 500) -> Dict[str, pd.DataFrame]:
        """Generate all datasets for the hackathon"""
        
        print(f"Generating {n_properties} commercial properties...")
        properties = self.generate_properties(n_properties)
        
        print("Generating tenant data...")
        tenants = self.generate_tenant_data(properties)
        
        print("Generating 24 months of historical metrics...")
        historical = self.generate_historical_metrics(properties, months=24)
        
        print("Generating market comparables...")
        comparables = self.generate_market_comparables(properties)
        
        print("Generating document references...")
        documents = self.generate_document_references(properties)
        
        # Create summary statistics
        summary = self._generate_summary_stats(properties)
        
        datasets = {
            'properties': properties,
            'tenants': tenants,
            'historical_metrics': historical,
            'comparables': comparables,
            'documents': documents,
            'summary': summary
        }
        
        print(f"\nDataset Generation Complete!")
        print(f"- Properties: {len(properties):,} records")
        print(f"- Tenants: {len(tenants):,} records")
        print(f"- Historical Metrics: {len(historical):,} records")
        print(f"- Comparables: {len(comparables):,} records")
        print(f"- Documents: {len(documents):,} records")
        
        return datasets
    
    def _generate_summary_stats(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """Generate portfolio summary statistics"""
        
        summary = {
            'Total Properties': len(properties_df),
            'Total Square Feet': f"{properties_df['total_sqft'].sum():,.0f}",
            'Total Value': f"${properties_df['property_value'].sum():,.0f}",
            'Average NOI': f"${properties_df['noi_annual'].mean():,.0f}",
            'Average Cap Rate': f"{properties_df['cap_rate'].mean():.2%}",
            'Portfolio Occupancy': f"{properties_df['occupancy_rate'].mean():.1%}",
            'Average Energy Star Score': f"{properties_df['energy_star_score'].mean():.0f}",
            'LEED Certified Properties': f"{properties_df['leed_certified'].sum()}",
            'Properties by Type': properties_df['property_type'].value_counts().to_dict(),
            'Properties by Market': properties_df['market'].value_counts().to_dict()
        }
        
        return pd.DataFrame([summary])
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame], output_dir: str = './cbre_data'):
        """Save all datasets to CSV files"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in datasets.items():
            if name != 'summary':  # Summary is just for display
                filepath = os.path.join(output_dir, f'{name}.csv')
                df.to_csv(filepath, index=False)
                print(f"Saved {name} to {filepath}")
        
        # Save metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'record_counts': {name: len(df) for name, df in datasets.items()},
            'schema': {name: list(df.columns) for name, df in datasets.items()}
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nAll datasets saved to {output_dir}/")


# Main execution
if __name__ == "__main__":
    # Initialize generator
    generator = CBREDataGenerator(seed=42)
    
    # Generate all datasets
    datasets = generator.generate_all_datasets(n_properties=500)
    
    # Display summary
    print("\n" + "="*60)
    print("PORTFOLIO SUMMARY")
    print("="*60)
    print(datasets['summary'].T.to_string())
    
    # Save to CSV files
    generator.save_datasets(datasets)
    
    # Display sample data
    print("\n" + "="*60)
    print("SAMPLE PROPERTIES (First 5)")
    print("="*60)
    print(datasets['properties'][['property_id', 'property_name', 'property_type', 
                                  'market', 'total_sqft', 'occupancy_rate', 
                                  'noi_annual', 'property_value']].head())