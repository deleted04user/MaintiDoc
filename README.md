# MaintiDoc

Prototype Streamlit de diagnostic documentaire pour le simulateur SAN-LAB 2000. Le [cadrage](data/Projet_IA_Generative_Multimodale_V2.pdf) décrit le projet. Le [manuel SAN](data/Manuel_Maintenance_SAN_Projet_Integrateur.pdf) fournit le corpus, les cinq schémas et les quatre interventions fictives.

## Démarrer sous Windows

Prévoir Python 3.12 ou 3.13, Tesseract avec les langues `fra` et `eng`, Ollama en cours d’exécution et plusieurs gigaoctets libres pour les dépendances et les modèles. L’indexation et la première recherche téléchargent les modèles FastEmbed.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
tesseract --list-langs
ollama pull qwen3-vl:2b
.\.venv\Scripts\python.exe -m maintidoc.retrieval
.\.venv\Scripts\python.exe -m streamlit run app.py
```

L’interface s’ouvre sur [localhost:8501](http://localhost:8501). Arrêter Streamlit avant de réindexer avec le mode Qdrant embarqué. Relancer la commande d’indexation après une révision du manuel. Les versions précédentes restent dans le stockage local, et seules les preuves de la dernière indexation terminée sont interrogées. L’extraction des tableaux est adaptée à la mise en page du manuel fourni.

`MAINTIDOC_DATA` déplace les données persistantes. `QDRANT_URL` permet d’utiliser un serveur Qdrant et `QDRANT_API_KEY` fournit sa clé si nécessaire. `OLLAMA_URL` et `OLLAMA_MODEL` changent le point d’accès ou le modèle vision. `OLLAMA_NUM_CTX` règle la taille du contexte, fixée à 4096 par défaut. `FASTEMBED_CACHE_PATH` déplace le cache des encodeurs.

## Les quatre étapes

1. **Stockage.** Deux collections Qdrant persistantes, pour les passages et pour les figures. Chaque preuve conserve le document, sa révision, sa page, son identifiant et l’empreinte du PDF.
2. **Recherche.** Les requêtes combinent vecteurs textuels et mots-clés. Une photo ajoute son OCR et sa similarité CLIP avec les figures. La fusion RRF fournit les candidats au reranker BGE multilingue. Chaque affirmation doit citer un extrait retrouvé et le passage cité doit l’étayer avant affichage.
3. **Orchestration.** LangChain prépare les prompts. LangGraph orchestre la recherche, les lectures MCP, la réponse, une nouvelle tentative si une citation est invalide et l’interruption pour approbation. Une question sur un voyant orange sans code déclenche une clarification.
4. **MCP.** Le serveur local en stdio expose `get_equipment_history`, `get_open_tickets` et `create_ticket`. Le bouton d’approbation autorise le contenu exact du brouillon pour la session en cours. SQLite conserve le ticket, son horodatage et l’identifiant de session. Un nouvel envoi du même brouillon retourne le ticket existant.

## Démonstration

- « Que signifie l’alerte AL-01 ? » La ligne de référence se trouve à la page 5.
- « Quelle est la cause du voyant orange ? » Ajouter une photo si disponible. MaintiDoc demande le code, le modèle et le contexte du panneau.
- « Consulte l’historique et prépare un ticket pour AL-01 sur HBA1. » Comparer le brouillon au manuel, puis le refuser ou approuver sa création. Demander ensuite « Quels sont les tickets ouverts ? »

Les tests utilisent le vrai PDF, Tesseract, Qdrant, LangGraph et le transport MCP. Les encodeurs et le générateur de réponses sont remplacés par des doublures déterministes.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
