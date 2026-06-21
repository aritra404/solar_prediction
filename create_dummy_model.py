import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Generate dummy training data matching the expected features
# 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'Hour', 'Day', 'Month'
np.random.seed(42)
n_samples = 1000

# Features
ambient_temp = np.random.uniform(20, 45, n_samples)
module_temp = ambient_temp + np.random.uniform(0, 20, n_samples)
irradiation = np.random.uniform(0, 1.2, n_samples)
hour = np.random.randint(0, 24, n_samples)
day = np.random.randint(1, 32, n_samples)
month = np.random.randint(1, 13, n_samples)

X = pd.DataFrame({
    'AMBIENT_TEMPERATURE': ambient_temp,
    'MODULE_TEMPERATURE': module_temp,
    'IRRADIATION': irradiation,
    'Hour': hour,
    'Day': day,
    'Month': month
})

# Make up a simple synthetic target (AC_POWER)
# Peaks mid-day, follows irradiation, and affected by temperature
# Daytime (roughly 6 to 18) has power
is_day = (hour > 5) & (hour < 19)
ac_power = np.zeros(n_samples)
ac_power[is_day] = irradiation[is_day] * 1000 - (module_temp[is_day] - 25) * 5
ac_power = np.clip(ac_power, 0, None)  # Ensure no negative power

y = ac_power

# Train a very simple Random Forest Regressor
model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'solar_power_model.pkl')
print("Successfully generated dummy solar_power_model.pkl")
