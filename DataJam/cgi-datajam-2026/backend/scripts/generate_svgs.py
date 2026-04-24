"""Generate SVG graphs from emergency department data.

Run this script from the backend directory after installing the required
packages (pandas, matplotlib, seaborn).

Outputs are saved to backend/graphs/ as SVG files.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# use seaborn styles
sns.set(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "hospital_service_volume", "Hospital_Service_Volumes_CTAS_Wide_20260306.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "graphs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# read the dataset
df = pd.read_csv(DATA_PATH, parse_dates=["Date"], dayfirst=False)

# convert Date column to period or datetime for plotting
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')

# 1. ER Demand by Hospital
avg_er = df.groupby('Hospital')['total_ER'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
avg_er.plot(kind='bar', color='steelblue')
plt.title('Average Monthly ER Visits by Hospital')
plt.ylabel('Average total_ER')
plt.xlabel('Hospital')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ER_demand_by_hospital.svg'), format='svg')
plt.clf()

# 2. Low-Acuity Visit Percentage by Hospital
avg_lowratio = df.groupby('Hospital')['low_ratio'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
avg_lowratio.plot(kind='bar', color='coral')
plt.title('Average Low-Acuity Visit Percentage by Hospital')
plt.ylabel('Average low_ratio')
plt.xlabel('Hospital')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'low_acuity_percentage_by_hospital.svg'), format='svg')
plt.clf()

# 3. CTAS Severity Distribution by Hospital (stacked bar)
severity_cols = ['CTAS1', 'CTAS2', 'CTAS3', 'CTAS4', 'CTAS5']
stack_df = df.groupby('Hospital')[severity_cols].sum()
stack_df = stack_df.sort_values(by=severity_cols, ascending=False)
plt.figure(figsize=(12, 8))
stack_df.plot(kind='bar', stacked=True, colormap='tab20', width=0.8)
plt.title('CTAS Severity Distribution by Hospital')
plt.ylabel('Number of Visits')
plt.xlabel('Hospital')
plt.xticks(rotation=45, ha='right')
plt.legend(title='CTAS Level')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ctas_severity_distribution.svg'), format='svg')
plt.clf()

# 4. Monthly ER Demand Trend
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='Date', y='total_ER', hue='Hospital', marker='o')
plt.title('Monthly ER Demand Trend by Hospital')
plt.ylabel('total_ER')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'monthly_er_trend.svg'), format='svg')
plt.clf()

# 5. Low-Acuity Demand Over Time
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='Date', y='low_severity', hue='Hospital', marker='o')
plt.title('Monthly Low-Acuity Demand by Hospital')
plt.ylabel('low_severity')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'low_acuity_trend.svg'), format='svg')
plt.clf()

# 6. Low-Acuity Share of ER Visits
# compute overall totals
totals = df[severity_cols].sum()
low = totals['CTAS4'] + totals['CTAS5']
high = totals['CTAS1'] + totals['CTAS2'] + totals['CTAS3']
plt.figure(figsize=(6, 6))
plt.pie([low, high], labels=['Low-acuity (CTAS4/5)', 'High-acuity (CTAS1-3)'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title('Low-Acuity Share of ER Visits (Overall)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'low_acuity_share.svg'), format='svg')
plt.clf()

print(f"SVG graphs written to {OUTPUT_DIR}")
