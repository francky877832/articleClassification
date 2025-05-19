import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Assurez-vous que ces lignes sont exécutées une fois
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialisation
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Nettoyage de texte
def clean_and_process_text(text):
    tokens = word_tokenize(str(text))
    cleaned_tokens = [
        lemmatizer.lemmatize(word.lower()) 
        for word in tokens 
        if word.lower() not in stop_words and word.isalpha()
    ]
    return " ".join(cleaned_tokens)

# Traitement par fichier individuel
def process_individual_files(input_files, output_folder, max_records=5000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in input_files:
        print(f"\nTraitement de : {file}")
        if os.path.exists(file):
            df = pd.read_csv(file)

            if 'summary' in df.columns and 'label' in df.columns:
                # Nettoyer la colonne summary
                df['summary_processed'] = df['summary'].apply(clean_and_process_text)

                # Prendre exactement 5000 échantillons (aléatoires)
                if len(df) >= max_records:
                    df_sampled = df.sample(n=max_records, random_state=42)
                else:
                    print(f"⚠️ {file} contient seulement {len(df)} données. Moins que 5000.")
                    continue

                # Sauvegarder dans un nouveau fichier
                base_name = os.path.splitext(os.path.basename(file))[0]
                output_path = os.path.join(output_folder, f"{base_name}_5000.csv")
                df_sampled.to_csv(output_path, index=False)

                # Afficher info
                label_name = df_sampled['label'].iloc[0]
                print(f"✅ {output_path} sauvegardé avec {len(df_sampled)} données. Label : {label_name}")
            else:
                print("❌ Colonnes 'summary' ou 'label' manquantes dans le fichier.")
        else:
            print(f"❌ Fichier {file} introuvable.")

# Fichiers à traiter
input_files = [
    'deep_learning_dataset.csv',
    'wireless_communication_dataset.csv',
    'llm_dataset.csv',
    'cloud_computing_dataset.csv',
    'virtual_reality_dataset.csv'
]

# Dossier de sortie
output_folder = 'processed_classes'

# Lancer le traitement
process_individual_files(input_files, output_folder)
