import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialiser les objets nécessaires
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Fonction de nettoyage et de traitement du texte
def clean_and_process_text(text):
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Enlever les stopwords et lemmatiser
    cleaned_tokens = [
        lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words and word.isalpha()
    ]
    
    return " ".join(cleaned_tokens)

# Fonction pour traiter et unifier les datasets
def unify_and_process_datasets(input_files, output_file, max_records=5000):
    all_data = []

    for file in input_files:
        print(f"Traitement du fichier : {file}")
        
        if os.path.exists(file):
            # Charger le fichier CSV
            df = pd.read_csv(file)
            
            # Appliquer le nettoyage et le traitement sur la colonne 'summary'
            if 'summary' in df.columns:
                df['summary_processed'] = df['summary'].apply(clean_and_process_text)
            
            # Limiter à 5000 lignes
            df = df.sample(n=min(len(df), max_records), random_state=42)  # Prendre jusqu'à 5000 lignes
            
            # Afficher le nombre de données récupérées par label
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                print(f"Nombre de données récupérées par label dans {file} :\n{label_counts}")
            
            # Ajouter les données traitées à la liste
            all_data.append(df)
        else:
            print(f"Le fichier {file} n'existe pas.")

    # Fusionner tous les DataFrames en un seul
    unified_df = pd.concat(all_data, ignore_index=True)
    
    # Supprimer les doublons basés sur la colonne 'url' (si nécessaire)
    if 'url' in unified_df.columns:
        unified_df.drop_duplicates(subset="url", inplace=True)
    
    # Sauvegarder le dataset unifié dans un fichier CSV
    unified_df.to_csv(output_file, index=False)
    print(f"Dataset unifié et traité sauvegardé dans {output_file}")

# Liste des fichiers d'entrée (fichiers CSV)
input_files = [
    'deep_learning_dataset.csv',
    'wireless_communication_dataset.csv',
    'llm_dataset.csv',
    'cloud_computing_dataset.csv',
    'virtual_reality_dataset.csv'
]

# Nom du fichier de sortie
output_file = 'unified_processed_dataset.csv'

# Unifier et traiter les datasets en limitant à 5000 lignes par fichier
unify_and_process_datasets(input_files, output_file, max_records=5000)
