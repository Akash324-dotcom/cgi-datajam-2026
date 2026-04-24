import pandas as pd

df = pd.read_csv("C:\\Users\\akash\\DataJam\\cgi-datajam-2026\\backend\\data\\processed\\hospital_service_volume\\Hospital_Service_Volumes_CTAS_Wide_20260306.csv")

# Extract month as numeric
df['Month'] = pd.to_datetime(df['Date'], format='%b-%y').dt.month

# Compute a simple proxy for expected wait time:
# Assume higher CTAS1+CTAS2 load increases wait
df['wait_score'] = (df['CTAS1']*3 + df['CTAS2']*2 + df['CTAS3'] + df['CTAS4']*0.5) / df['total_ER']