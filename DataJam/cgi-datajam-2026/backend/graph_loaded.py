import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data
data = pd.read_csv(".\\data\\processed\\hospital_service_volume\\Hospital_Service_Volumes_CTAS_Wide_20260306.csv")  # replace with your CSV file path

# Calculate low-severity patients (CTAS 4 + 5)
data['Low_Severity'] = data['CTAS4'] + data['CTAS5']

# Optionally calculate the ratio of low-severity patients
data['Low_Ratio'] = data['Low_Severity'] / data['total_ER']

# Convert 'Date' to datetime for proper plotting
data['Date'] = pd.to_datetime(data['Date'], format='%b-%y')

# Plotting
plt.figure(figsize=(12,6))

for hospital in data['Hospital'].unique():
    hospital_data = data[data['Hospital'] == hospital]
    plt.plot(hospital_data['Date'], hospital_data['Low_Severity'], marker='o', label=f"{hospital} - Low Severity")

plt.plot(data['Date'], data['total_ER'], marker='x', linestyle='--', color='black', label="Total ER Visits")

plt.title("Low-Severity ER Visits (CTAS 4-5) vs Total ER Visits")
plt.xlabel("Date")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()