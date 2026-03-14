import pandas as pd
import os

# Using the exact name from your screenshot sidebar
source_file = "phishing-sites_url.csv"
output_file = "dataset.csv"

if not os.path.exists(source_file):
    print(f"❌ Error: {source_file} not found in this folder!")
else:
    print("✅ Loading your data...")
    # Your screenshot shows columns 'URL' and 'Label'
    df = pd.read_csv(source_file)

    print("🔄 Converting labels...")
    # Convert 'bad' to 1 and 'good' to 0
    df['Label'] = df['Label'].str.lower().map({'good': 0, 'bad': 1})

    # Remove any empty rows and rename for your trainer
    df = df.dropna(subset=['Label'])
    df = df.rename(columns={'URL': 'url', 'Label': 'label'})
    df['label'] = df['label'].astype(int)

    # Save it
    df[['url', 'label']].to_csv(output_file, index=False)
    print(f"✔️ Success! Created '{output_file}' with {len(df)} samples.")