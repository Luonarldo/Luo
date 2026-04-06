import os
import pandas as pd

columns_to_sum = {
    '道路':['Curb',  
          'Crosswalk - Plain', 
          'Curb Cut', 
          'Lane Marking - Crosswalk', 
          'Lane Marking - General', 
          'Manhole',
          'Ego Vehicle',
          'Road',
          'Service Lane',
          'Catch Basin'],
    '建筑':['Billboard', 'Building'],
    '车辆':['Bus', 'Car', 'Caravan', 'Other Vehicle', 'Truck'],
}

in_csv = '../tw/image-output/result.csv'
out_csv = '../tw/image-output/merged.xlsx'

df = pd.read_csv(in_csv)
for category, cols in columns_to_sum.items():
    df[category] = df[cols].sum(axis=1)
df.to_excel(out_csv, index=False)
# out_excel = os.path.splitext(out_csv)[0] + '.xlsx'
# df.to_excel(out_excel, index=False)