import scholarly
import pandas as pd
import time

# Fonction de récupération des articles via Google Scholar
def scrape_google_scholar(query, label, max_results=1000):
    search_query = scholarly.search_pubs(query)
    results = []
    total_scraped = 0

    # Récupérer les articles par lots
    for i in range(max_results // 10):  # Récupère par paquet de 10 articles
        try:
            print(f"[{label}] Scraping articles: {i * 10 + 1}–{(i + 1) * 10}")
            article = next(search_query)

            title = article['bib']['title']
            summary = article['bib'].get('abstract', 'No summary available')
            url = article['url_scholarbib']  # URL vers l'article

            results.append({
                "title": title,
                "summary": summary,
                "url": url,
                "label": label
            })

            total_scraped += 1
            time.sleep(1)  # Délai pour éviter d'être bloqué

        except StopIteration:
            break  # Arrêter si plus d'articles disponibles
        except Exception as e:
            print(f"Erreur lors du scraping : {e}")

    # Sauvegarder les résultats dans un fichier CSV
    if results:
        df_new = pd.DataFrame(results)
        filename = f"{label.lower().replace(' ', '_')}_google_scholar.csv"
        if os.path.exists(filename):
            df_existing = pd.read_csv(filename)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset="url", inplace=True)
        else:
            df_combined = df_new
        df_combined.to_csv(filename, index=False)
        print(f"[{label}] Scraping terminé. Total articles : {len(df_combined)}")
    else:
        print(f"[{label}] Aucun article trouvé.")

# Ajouter les articles manquants pour chaque classe (ici pour LLM)
scrape_google_scholar(query="large language model", label="LLM", max_results=5000)
