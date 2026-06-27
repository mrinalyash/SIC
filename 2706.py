import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
import requests

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\NFHS-5-States.csv")

pivot_df = df.pivot(index='state', columns='indicator', values='nfhs5_total').reset_index()

pivot_df = pivot_df[['state',
    '92. Children age 6-59 months who are anaemic (<11.0 g/dl)22 (%)',
    '95. All women age 15-49 years who are anaemic22 (%)',
    '107. Elevated blood pressure (Systolic ≥140 mm of Hg and/or Diastolic ≥90 mm of Hg) or taking medicine to control blood pressure (%)',
    '101. Blood sugar level - high or very high (>140 mg/dl) or taking medicine to control blood sugar level23 (%)',
    '97. Men age 15-49 years who are anaemic (<13.0 g/dl)22 (%)',
    '89. Men who are overweight or obese (BMI ≥25.0 kg/m2) (%)',
    '93. Non-pregnant women age 15-49 years who are anaemic (<12.0 g/dl)22 (%)',
    '88. Women who are overweight or obese (BMI ≥25.0 kg/m2)21 (%)'
]]

pivot_df.columns = ['state', 'Children_Anaemia', 'All_Women_Anaemia', 'Elevated_Blood_Pressure',
                    'High_Blood_Sugar', 'Men_Anaemia', 'Men_Overweight_Obese',
                    'NonPregnant_Women_Anaemia', 'Women_Overweight_Obese']

for col in pivot_df.columns:
    if col != 'state':
        pivot_df[col] = pd.to_numeric(pivot_df[col], errors='coerce')

pivot_df['Gender_Obesity_Gap'] = (pivot_df['Women_Overweight_Obese'] - pivot_df['Men_Overweight_Obese']).round(1)

# === Task 1: Correlation Matrix ===
print("=== Task 1: Correlation Matrix ===")

key_cols = ['Children_Anaemia', 'Women_Overweight_Obese', 'Men_Overweight_Obese', 
            'All_Women_Anaemia', 'High_Blood_Sugar', 'Elevated_Blood_Pressure']

corr_matrix = pivot_df[key_cols].corr(method='pearson')
print(corr_matrix.round(3))
print(f"\nChildren Anaemia vs Women Obesity: {pivot_df['Children_Anaemia'].corr(pivot_df['Women_Overweight_Obese']):.4f}")
print(f"Children Anaemia vs Men Obesity: {pivot_df['Children_Anaemia'].corr(pivot_df['Men_Overweight_Obese']):.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title("Correlation Matrix - Anaemia & Obesity/ Hypertension")
plt.tight_layout()
plt.show()

state_df = pivot_df[pivot_df['state'] != 'India'].copy()
top10 = state_df.nlargest(10, 'Elevated_Blood_Pressure')[['state', 'Elevated_Blood_Pressure']]

plt.figure(figsize=(12, 7))
x = range(len(top10))
plt.bar(x, top10['Elevated_Blood_Pressure'], width=0.6, label='Elevated Blood Pressure (%)', color='skyblue')
plt.xlabel('States')
plt.ylabel('Percentage (%)')
plt.title('Top 10 States with Highest Hypertension Rates')
plt.xticks(x, top10['state'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(state_df['All_Women_Anaemia'], state_df['Women_Overweight_Obese'], s=100, alpha=0.7, color='teal')
for i, row in state_df.iterrows():
    label = row['state'][:12] + '..' if len(row['state']) > 12 else row['state']
    plt.text(row['All_Women_Anaemia'] + 0.5, row['Women_Overweight_Obese'] + 0.5, label, fontsize=9)
plt.xlabel('Anaemic Women (%)')
plt.ylabel('Obese Women (%)')
plt.title('Anaemic Women vs Obese Women by State')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

pivot_df.to_csv('nfhs5_clean_data.csv', index=False)
geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"

geojson_data = requests.get(geojson_url).json()
pivot_df['state'] = pivot_df['state'].replace({
    'Andaman & Nicobar Islands':'Andaman and Nicobar Islands',
    'Dadra & Nagar Haveli and Daman & Diu':
        'Dadra and Nagar Haveli and Daman and Diu',
    'NCT of Delhi':'Delhi',
    'Jammu & Kashmir':'Jammu and Kashmir',
    'Odisha':'Odisha'
})
india_map = folium.Map(
    location=[20.5937, 78.9629],
    zoom_start=5,
    tiles='cartodbpositron'
)
folium.Choropleth(
    geo_data=geojson_data,
    data=pivot_df,
    columns=['state','Women_Overweight_Obese'],
    key_on='feature.properties.NAME_1',
    fill_color='YlOrRd',
    fill_opacity=0.8,
    line_opacity=0.3,
    legend_name='Women Obesity (%)'
).add_to(india_map)
state_capitals = {
    "Andhra Pradesh": [16.5062, 80.6480],
    "Arunachal Pradesh": [27.0844, 93.6053],
    "Assam": [26.1445, 91.7362],
    "Bihar": [25.5941, 85.1376],
    "Chhattisgarh": [21.2514, 81.6296],
    "Goa": [15.4909, 73.8278],
    "Gujarat": [23.2156, 72.6369],
    "Haryana": [30.7333, 76.7794],
    "Himachal Pradesh": [31.1048, 77.1734],
    "Jharkhand": [23.3441, 85.3096],
    "Karnataka": [12.9716, 77.5946],
    "Kerala": [8.5241, 76.9366],
    "Madhya Pradesh": [23.2599, 77.4126],
    "Maharashtra": [19.0760, 72.8777],
    "Manipur": [24.8170, 93.9368],
    "Meghalaya": [25.5788, 91.8933],
    "Mizoram": [23.7271, 92.7176],
    "Nagaland": [25.6751, 94.1086],
    "Odisha": [20.2961, 85.8245],
    "Punjab": [30.7333, 76.7794],
    "Rajasthan": [26.9124, 75.7873],
    "Sikkim": [27.3389, 88.6065],
    "Tamil Nadu": [13.0827, 80.2707],
    "Telangana": [17.3850, 78.4867],
    "Tripura": [23.8315, 91.2868],
    "Uttar Pradesh": [26.8467, 80.9462],
    "Uttarakhand": [30.3165, 78.0322],
    "West Bengal": [22.5726, 88.3639],
    "Delhi": [28.6139, 77.2090],
    "Puducherry": [11.9416, 79.8083],
    "Chandigarh": [30.7333, 76.7794]
}
for _, row in pivot_df.iterrows():

    state = row["state"]

    if state in state_capitals:

        popup_html = f"""
        <div style="width:220px;font-family:Arial;">
        <h4 style="margin-bottom:5px;">{state}</h4>
        <hr>
        <b>High Blood Sugar</b><br>
        {row['High_Blood_Sugar']} %<br><br>

        <b>Women Anaemia</b><br>
        {row['All_Women_Anaemia']} %<br><br>

        <b>Women Obesity</b><br>
        {row['Women_Overweight_Obese']} %
        </div>
        """

        folium.CircleMarker(

            location=state_capitals[state],

            radius=6,

            color="darkred",

            weight=1,

            fill=True,

            fill_color="red",

            fill_opacity=0.85,

            popup=folium.Popup(popup_html, max_width=250),

            tooltip=state

        ).add_to(india_map)
        
india_map.save("India_Public_Health_Dashboard.html")

print("Interactive map saved successfully.")