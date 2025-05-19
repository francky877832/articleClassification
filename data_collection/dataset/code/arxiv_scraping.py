import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd

def clean_text(text):
    return (
        text.replace('\n', ' ')
            .replace('\r', ' ')
            .replace('\t', ' ')
            .replace('  ', ' ')
            .strip()
    )

def scrape_arxiv(keywords, label, max_results_total=10000):
    base_url = "http://export.arxiv.org/api/query?"
    results = []

    filename = f"{label.lower().replace(' ', '_')}_dataset.csv"
    existing_urls = set()

    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        existing_urls = set(existing_df["url"].dropna().tolist())
    else:
        existing_df = pd.DataFrame(columns=["title", "summary", "url", "label"])

    total_scraped = 0
    max_results_per_keyword = max_results_total // len(keywords)

    for keyword in keywords:
        print(f"\n🔍 [{label}] Mot-clé : '{keyword}'")
        for start in range(0, max_results_per_keyword, 100):
            print(f"[{label}] Fetching {start}–{start + 100} pour '{keyword}'")
            url = f"{base_url}search_query=all:{keyword}&start={start}&max_results=100"
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Erreur HTTP: {response.status_code}")
                break

            root = ET.fromstring(response.content)
            added = 0

            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
                link = entry.find('{http://www.w3.org/2005/Atom}id').text.strip()

                if link in existing_urls:
                    continue

                cleaned_title = clean_text(title)
                cleaned_summary = clean_text(summary)

                if len(cleaned_summary) < 200:
                    continue

                results.append({
                    "title": cleaned_title,
                    "summary": cleaned_summary,
                    "url": link,
                    "label": label
                })
                existing_urls.add(link)
                added += 1
                total_scraped += 1

                if total_scraped >= max_results_total:
                    print(f"[{label}] ✅ Objectif de {max_results_total} atteint.")
                    break

            print(f"[{label}] +{added} articles ajoutés. Total : {total_scraped}")

            if total_scraped >= max_results_total or added == 0:
                break

        if total_scraped >= max_results_total:
            break

    if results:
        df_new = pd.DataFrame(results)
        df_combined = pd.concat([existing_df, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset="url", inplace=True)
        df_combined.to_csv(filename, index=False)
        print(f"\n✅ [{label}] Dataset mis à jour : {len(df_combined)} articles")
    else:
        print(f"\n⚠️ [{label}] Aucun nouvel article ajouté.")

# ✅ Mots-clés enrichis pour chaque classe
classes = {
    "LLM": [
        "large language model", "chatgpt", "gpt-3", "gpt-4", "bert", 
        "llama", "transformer language model", "instruction tuning", "prompt engineering"
    ],
    "Deep Learning": [
        "deep learning", "convolutional neural network", "cnn", "rnn", "lstm",
        "transformer", "deep neural network", "autoencoder", "gan", "resnet"
    ],
    "Wireless Communication": [
        "wireless communication", "5g", "6g", "lte", "mimo", 
        "beamforming", "cognitive radio", "ofdm", "massive mimo", "wireless sensor network"
    ],
    "Cloud Computing": [
        "cloud computing", "serverless computing", "cloud storage", "cloud architecture", 
        "distributed computing", "virtual machines", "kubernetes", "docker", 
        "cloud infrastructure", "edge computing"
    ],
    "Virtual Reality": [
        "virtual reality", "augmented reality", "mixed reality", "vr headset", 
        "immersive environment", "metaverse", "haptic feedback", "3d interaction", 
        "vr training", "virtual simulation"
    ]
}

# 🚀 Lancer le scraping
for label, keywords in classes.items():
    print(f"\n🚀 Scraping pour la classe : {label}")
    scrape_arxiv(keywords=keywords, label=label, max_results_total=10000)
