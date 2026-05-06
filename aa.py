import requests

url = "https://tech-challenge-churn.onrender.com/predict"

payload = {
    "gender": "Male",               
    "senior_citizen": "No",         
    "partner": "No",                
    "dependents": "No",             
    "tenure_months": 24,            # era "tenure", corrigido para "tenure_months"
    "monthly_charges": 75.5,
    "total_charges": 1800.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "tech_support": "No",
    "paperless_billing": "Yes",     # era "true", corrigido para "Yes"
    "payment_method": "Electronic check",
    "phone_service": "Yes",         
    "multiple_lines": "No",         # "Yes", "No" ou "No phone service"
    "online_security": "No",        
    "online_backup": "No",          
    "device_protection": "No",      
    "streaming_tv": "No",           
    "streaming_movies": "No",       
}
response = requests.post(url, json=payload)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")