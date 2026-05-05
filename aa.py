import requests

url = "https://tech-challenge-churn.onrender.com/predict"

payload = {
  "tenure": 24,
  "monthly_charges": 75.5,
  "total_charges": 1800.0,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic",
  "tech_support": "No",
  "paperless_billing": "true",
  "payment_method": "Electronic check"
}

response = requests.post(url, json=payload)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")