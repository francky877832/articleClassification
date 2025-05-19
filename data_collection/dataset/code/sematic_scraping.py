import requests
import pandas as pd
import time
import os

def scrape_semantic_scholar(query, label, max_results=10000):
    print(f"\n🔍 Scraping Semantic Scholar pour la classe : '{label}' | Max : {max_results}")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    batch_size = 100
    offset = 0
    total = 0
    all_results = []

    filename = f"{label.lower().replace(' ', '_')}_semantic_scholar.csv"
    existing_urls = set()

    # Charger les URLs déjà existantes
    if os.path.exists(filename):
        print("📁 Lecture des données existantes...")
        df_existing = pd.read_csv(filename)
        existing_urls = set(df_existing['url'].dropna().unique())
        print(f"🔎 {len(existing_urls)} articles déjà présents.")

    while total < max_results:
        params = {
            "query": query,
            "limit": batch_size,
            "offset": offset,
            "fields": "title,abstract,url"
        }

        try:
            res = requests.get(url, params=params, headers=headers, timeout=10)
            res.raise_for_status()
            data = res.json().get("data", [])
        except Exception as e:
            print(f"⚠️ Erreur à l’offset {offset} : {e}")
            break

        if not data:
            print("🚫 Aucune donnée reçue. Fin.")
            break

        new_articles = 0
        for paper in data:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            paper_url = paper.get("url", "")

            if title and abstract and paper_url not in existing_urls:
                all_results.append({
                    "title": title,
                    "summary": abstract,
                    "url": paper_url,
                    "label": label
                })
                existing_urls.add(paper_url)
                new_articles += 1

        print(f"📦 Batch [{offset}–{offset + batch_size}] : {new_articles} nouveaux | Total : {len(existing_urls)}/{max_results}")
        offset += batch_size
        total = len(existing_urls)

        if new_articles == 0:
            print("✅ Aucun nouvel article trouvé dans ce batch. On stoppe.")
            break

        time.sleep(1)  # Respecter le serveur

    # Sauvegarde
    if all_results:
        df_new = pd.DataFrame(all_results)
        if os.path.exists(filename):
            df_existing = pd.read_csv(filename)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset="url", inplace=True)
        else:
            df_combined = df_new

        df_combined.to_csv(filename, index=False)
        print(f"\n✅ Fichier mis à jour : {filename} ({len(df_combined)} articles au total)\n")
    else:
        print("⚠️ Aucun nouvel article n’a été récupéré.")

# Exemple : scrape 10k articles sur LLM
#scrape_semantic_scholar("large language model", "LLM", max_results=10000)


cloud_queries = [
    "cloud architecture",
    "cloud storage",
    "cloud scalability",
    "cloud security",
    "cloud orchestration",
    "cloud platform",
    "cloud-native applications",
    "virtualization",
    "containerization",
    "microservices",
    "infrastructure as a service",
    "platform as a service",
    "software as a service",
    "edge computing",
    "fog computing"
]


for i, q in enumerate(cloud_queries):
    scrape_semantic_scholar(q, "Cloud Computing", max_results=6000)

llm_queries = [
    "large language model",
    "LLM NLP",
    "transformer model",
    "transformer language model",
    "GPT model",
    "BERT model",
    "T5 model",
    "text generation model",
    "language generation",
    "autoregressive model",
    "prompt learning",
    "instruction tuning",
    "pretrained language model",
    "multilingual language model",
    "foundation model",
    "fine-tuned language model",
    "language model for reasoning",
    "causal language model"
]
for q in llm_queries:
    scrape_semantic_scholar(q, "LLM", max_results=6000)


dl_queries = [
    "deep learning",
    "convolutional neural networks",
    "recurrent neural networks",
    "autoencoders",
    "deep neural networks",
    "deep reinforcement learning"
]

for i, q in enumerate(dl_queries):
    scrape_semantic_scholar(q, "Deep Learning", max_results=6000)

