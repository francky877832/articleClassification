# Liste des mots-clés pour LLM (Large Language Model)
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

# Remplacer une seule entrée dans 'classes' par les mots-clés LLM
classes = {
    "LLM": llm_queries,  # Utiliser la liste de requêtes pour LLM
}


# Fonction pour scraper ArXiv avec plusieurs mots-clés pour chaque catégorie
def scrape_arxiv(query_list, label, max_results=10000):
    base_url = "http://export.arxiv.org/api/query?"
    results = []

    # Fichier CSV pour chaque classe
    filename = f"{label.lower().replace(' ', '_')}_dataset.csv"
    existing_urls = set()

    # Chargement du dataset existant pour éviter les doublons
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        existing_urls = set(existing_df["url"].dropna().tolist())
    else:
        existing_df = pd.DataFrame(columns=["title", "summary", "url", "label"])

    total_scraped = 0

    # Boucle sur chaque mot-clé de la liste
    for query in query_list:
        print(f"\n[{label}] Scraping pour le mot-clé : '{query}'")
        for start in range(0, max_results, 100):
            print(f"[{label}] Fetching {start}–{start + 100}")
            url = f"{base_url}search_query=all:{query}&start={start}&max_results=100"
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
                    continue  # Doublon

                cleaned_title = clean_text(title)
                cleaned_summary = clean_text(summary)

                # Vérifier si le résumé est trop court
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

            print(f"[{label}] +{added} nouveaux articles ajoutés pour '{query}'. Total : {total_scraped}")

            if total_scraped >= max_results:
                print(f"[{label}] Atteint la limite des {max_results} articles.")
                break  # Arrêter si on a atteint la limite

    # Sauvegarder les résultats dans un fichier CSV
    if results:
        df_new = pd.DataFrame(results)
        df_combined = pd.concat([existing_df, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset="url", inplace=True)
        df_combined.to_csv(filename, index=False)
        print(f"[{label}] ✅ Dataset mis à jour : {len(df_combined)} articles")
    else:
        print(f"[{label}] Aucun nouvel article à ajouter (doublons ou trop courts).")

# Scraping pour LLM avec plusieurs mots-clés
scrape_arxiv(llm_queries, "LLM", max_results=5000)
