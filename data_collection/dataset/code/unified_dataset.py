import pandas as pd
import os

# Dossier où sont stockés les fichiers traités (5000 par label)
folder = 'processed_classes'  # change si tu as un autre nom

# Liste pour stocker tous les DataFrames
all_data = []

# Lister et traiter tous les fichiers .csv dans le dossier
for file in os.listdir(folder):
    if file.endswith(".csv"):
        filepath = os.path.join(folder, file)
        print(f"📥 Chargement du fichier : {file}")
        df = pd.read_csv(filepath)
        all_data.append(df)

# Concaténer tous les DataFrames en un seul
unified_df = pd.concat(all_data, ignore_index=True)

# Enregistrer le fichier final
output_file = 'unified_final_dataset.csv'
unified_df.to_csv(output_file, index=False)

print(f"\n✅ Dataset unifié enregistré dans : {output_file}")
print(f"🧾 Total d'exemples : {len(unified_df)}")
