# **🤖 pyAI**
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code Style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-4B8BBE.svg)](https://peps.python.org/pep-0008/)

**Un modèle de langage léger de type GPT (décodeur-only Transformer) implémenté de zéro en Python.**
Entraînez votre propre modèle de langage sur un corpus personnalisé, avec tokenization BPE (SentencePiece), entraînement optimisé (AdamW + Cosine LR Scheduler), et génération de texte autorégressive.

Ce projet est avant tout un projet personnel, une base pour comprendre comment fonctionne un GPT. Si vous souhaitez d"couvrir l'IA je vous incite à aller voir 
la chaîne youtube de [CodeWithAarohi](https://www.youtube.com/@CodeWithAarohi/videos)

---

## **Table des matières**
- [Aperçu](#-aperçu)
- [Architecture](#-architecture)
- [Fonctionnalités](#-fonctionnalités)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
  - [Entraînement d'un modèle](#entraînement-dun-modèle)
  - [Génération de texte](#génération-de-texte)
  - [Dialogue avec le modèle](#dialogue-avec-le-modèle)
- [Structure du projet](#-structure-du-projet)
- [Configuration](#-configuration)
- [Limitations](#-limitations)
- [Améliorations possibles](#-améliorations-possibles)
- [Contributions](#-contributions)

---

## **Aperçu**
**pyAI** est une implémentation **minimale mais fonctionnelle** d'un modèle de langage inspiré de **GPT** (Generative Pre-trained Transformer).
Contrairement aux grands modèles comme GPT-3 ou Llama, ce projet est conçu pour :
- **Être compréhensible** : Code commenté et structuré pour l'apprentissage.
- **Être léger** : Entraînement possible sur un **GPU grand public** (voire un CPU pour de petits corpus).
- **Être personnalisable** : Adapté à vos propres jeux de données.

> ✨ **Inspiré par** : [codewithaarohi/Build_mini_gpt_with_tokenizer](https://github.com/codewithaarohi/Build_mini_gpt_with_tokenizer)

---

---

## **Architecture**
### **Tokenizer (SentencePiece BPE)**
- **Type** : Byte Pair Encoding (BPE) via [SentencePiece](https://github.com/google/sentencepiece).
- **Vocabulaire** : Par défaut **115 tokens** (configurable).
- **Avantages** :
  - Gestion native des mots inconnus (subword tokenization).
  - Optimisé pour les langues sans espaces (ex: chinois, japonais).

### **Modèle Transformer (Décodeur-only)**
| Composant               | Implémentation                          | Détails                                                                 |
|-------------------------|-----------------------------------------|-------------------------------------------------------------------------|
| **Embeddings**          | `nn.Embedding`                          | Tokens + positions (taille configurable).                            |
| **Blocs Transformer**   | `TransformerBlock` (custom)             | Attention multi-têtes + Feed-Forward Network + LayerNorm.              |
| **Tête de sortie**      | `nn.Linear`                             | Prédit le prochain token (vocabulaire → logits).                     |
| **Fonction d'attention**| Attention standard (sans mask causal)   | **⚠️ À corriger** : Actuellement, le modèle voit les tokens futurs. |

### **Pipeline d'entraînement**
- **Optimiseur** : [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) (avec `weight_decay=1e-2`).
- **Scheduler** : [CosineAnnealingLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) (diminution progressive du learning rate).
- **Gradient Clipping** : `max_norm=1.0` (évite l'explosion des gradients).
- **Split données** : 90% entraînement / 10% validation.
- **Early Stopping** : Arrêt si aucune amélioration après `patience` itérations.

### **Génération de texte**
- **Méthode** : Sampling aléatoire (`torch.multinomial` + softmax).
- **Limitations** :
  - Pas de **temperature** ou **top-k sampling** (à implémenter).
  - Pas de **beam search** (génération basique).

---

---

## **Fonctionnalités**
| Fonctionnalité          | Description                                                                 | Statut          |
|------------------------|-----------------------------------------------------------------------------|-----------------|
| Tokenization BPE       | Entraînement d'un tokenizer SentencePiece sur un corpus personnalisé.     | ✅ Fonctionnel   |
| Entraînement            | Boucle complète avec validation périodique.                                | ✅ Fonctionnel   |
| Sauvegarde/Chargement   | Stockage des poids du modèle au format `.pth`.                             | ✅ Fonctionnel   |
| Génération de texte    | Prédiction autorégressive de tokens.                                      | ✅ Fonctionnel   |
| Interface de dialogue   | Discussion simple avec le modèle via `chat()`.                            | ✅ Fonctionnel   |
| Mask causal            | Empêcher le modèle de voir les tokens futurs.                              | ❌ **À implémenter** |
| Mixed Precision        | Accélération de l'entraînement avec `torch.cuda.amp`.                    | ❌ **À implémenter** |
| DataLoader PyTorch     | Batch optimisé avec shuffling et parallélisation.                         | ❌ **À implémenter** |

---

## **Prérequis**
| Outil               | Version requise       | Lien                                                                 |
|---------------------|------------------------|----------------------------------------------------------------------|
| Python              | 3.8+                  | [python.org](https://www.python.org/downloads/)                     |
| PyTorch             | 2.0+                  | [pytorch.org](https://pytorch.org/)                                |
| SentencePiece       | 0.1.97+               | `pip install sentencepiece`                                         |
| CUDA (optionnel)    | 11.8+ (pour GPU)      | [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)   |

> 💡 **Note** : Si vous n'avez pas de GPU, le modèle s'exécutera sur CPU (plus lent).

---

---

## **Installation**
### **1️ Cloner le dépôt**
```bash
git clone https://github.com/votre-utilisateur/KIPP-AI.git
cd KIPP-AI
```

### **2️ Installer les dépendances**
```bash
pip install torch sentencepiece
```

### **3️ Vérifier l'installation**
```python
import torch
import sentencepiece as spm
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"SentencePiece version: {spm.__version__}")
```

---

## **Utilisation**
### **Préparation des données**
1. Créez un dossier pour votre projet (ex: `data/`).
2. Placez-y votre **corpus texte** (ex: `corpus.txt`).
   - Format : **Texte brut** (un token par ligne ou paragraphe).
   - Exemple :
     ```
     Bonjour, comment ça va ?
     Je vais bien, merci !
     Aujourd'hui, il fait beau.
     ```

---

### **Initialisation du modèle**
```python
from pyAI import GPT

# Initialiser le modèle avec le chemin vers votre dossier de données
model = GPT(source_dir="C:/chemin/vers/votre/dossier/").to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
```

---

### **Entraînement d'un modèle**
```python
# 1. Charger un corpus
model.load_corpus(corpus_name="corpus.txt")

# 2. Créer un nouveau modèle (optionnel)
model.create_model(model_name="mon_modele")

# 3. Charger un modèle existant (ou créer s'il n'existe pas)
model.load_model(model_name="mon_modele")

# 4. Entraîner le modèle
model.train(
    convergence_threshold=0.2,  # Arrêt si la perte val < 0.2
    eval_interval=100,          # Évaluation toutes les 100 étapes
    patience=1000               # Early stopping après 1000 itérations sans amélioration
)
```
>  **Avertissement** :
> - Si `convergence_threshold < 0.1`, l'entraînement peut ne **jamais s'arrêter**.
> - Pour un entraînement rapide, utilisez un **petit corpus** (ex: 1 Mo de texte).

---

### **Génération de texte**
```python
# Générer 20 tokens à partir d'une phrase de départ
input_text = "Bonjour, comment"
user_input = torch.tensor([model.tokenizer.encode(input_text)], dtype=torch.long).to(model.device)
output = model.generate(user_input, num_new_tokens=20)
generated_text = model.tokenizer.decode(output[0].tolist())
print(generated_text)
```

---
### **Dialogue avec le modèle**
```python
# Discuter avec le modèle (interface simplifiée)
model.chat(text="Bonjour, comment ça va ?", num_new_tokens=20)
```
**Exemple de sortie :**
```
Bonjour, comment ça va ? Je vais très bien, merci de demander !
```

---
---
---

## **Structure du projet**
```text
KIPP-AI/
├── pyAI.py               # Code principal du modèle GPT
├── TransformerBlock.py   # Implémentation des blocs Transformer
├── README.md             # Documentation (ce fichier)
├── data/                 # Dossier pour les corpus et modèles (à créer)
│   ├── corpus.txt        # Exemple de corpus texte
│   ├── tokenizer.model   # Tokenizer SentencePiece (généré)
│   └── mon_modele.pth    # Poids du modèle (généré)
└── requirements.txt      # Dépendances Python
```

---
---
---

## **Configuration**
Vous pouvez modifier les **hyperparamètres** du modèle directement dans `pyAI.py` :

| Paramètre            | Variable dans `GPT.__init__` | Valeur par défaut | Recommandation                     |
|----------------------|-------------------------------|-------------------|------------------------------------|
| Taille des blocs     | `block_size`                  | 64                | 128-256 pour de meilleurs résultats |
| Dimension embeddings | `embedding_dim`               | 256               | 512+ pour des modèles plus puissants |
| Nombre de têtes      | `num_heads`                   | 8                 | Doit diviser `embedding_dim`        |
| Nombre de couches    | `num_layers`                  | 6                 | 12+ pour des performances avancées |
| Learning rate        | `learning_rate`               | 1e-3              | 3e-4 à 5e-4 pour la stabilité      |
| Taille vocabulaire   | `vocab_size` (dans `load_corpus`) | 115           | 512-1024 pour du texte général       |

---

## **Limitations**
| Limitation                          | Impact                                                                 | Solution proposée                          |
|------------------------------------|------------------------------------------------------------------------|--------------------------------------------|
| **Pas de mask causal**              | Le modèle voit les tokens futurs → **triche** pendant l'entraînement. | Ajouter un mask triangulaire dans `TransformerBlock`. |
| **Pas de mixed precision**         | Entraînement lent sur GPU.                                             | Utiliser `torch.cuda.amp`.                  |
| **Pas de DataLoader**              | Batch non optimisé.                                                    | Remplacer `get_batch` par `DataLoader`.     |
| **Génération basique**             | Sampling aléatoire sans contrôle.                                     | Ajouter `temperature`, `top-k`, `beam search`. |
| **Pas de padding**                 | Batches de taille fixe uniquement.                                    | Implémenter le padding dynamique.          |
| **Tokenization re-entraînée**      | Lent si `load_corpus` est appelé plusieurs fois.                      | Sauvegarder le tokenizer une fois pour toutes. |

---

## **Améliorations possibles**
### **Priorité haute**
-  **Ajouter un mask causal** dans `TransformerBlock` (critique pour la génération).
-  **Corriger les bugs restants**.

---

## **Contributions**
Les contributions sont les bienvenues ! Voici comment contribuer :

### **Signaler un bug**
1. Vérifiez que le bug n'est pas déjà signalé dans les [issues](https://github.com/votre-utilisateur/KIPP-AI/issues).
2. Ouvrez une nouvelle issue avec :
   - Une **description claire** du problème.
   - Les **étapes pour reproduire** le bug.
   - Votre **environnement** (Python, PyTorch, OS).

### **Proposer une amélioration**
1. Forkez le dépôt.
2. Créez une branche (`git checkout -b feature/ma-fonctionnalité`).
3. Commitez vos modifications (`git commit -m "Ajout de ..."`).
4. Poussez vers la branche (`git push origin feature/ma-fonctionnalité`).
5. Ouvrez une **Pull Request** en décrivant vos changements.

### **Améliorer la documentation**
- Corriger les fautes d'orthographe.
- Ajouter des exemples d'utilisation.
- Traduire en anglais.

---

## **Remerciements et Crédits**
- **codewithaarohi** pour l'inspiration initiale ([Build_mini_gpt_with_tokenizer](https://github.com/codewithaarohi/Build_mini_gpt_with_tokenizer)).
- **PyTorch** et **SentencePiece** pour leurs bibliothèques open-source.


---

## **Contact**
Pour toute question ou suggestion, ouvrez une **issue** ou contactez-moi via :
- **Email** : [votre@email.com](mailto:votre@email.com)
- **GitHub** : [@votre-utilisateur](https://github.com/votre-utilisateur)

---

> **💡 Saviez-vous ?**
> Ce projet a été créé pour **apprendre** comment fonctionnent les modèles de langage.
> Si vous voulez aller plus loin, explorez :
> - [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) (pour des modèles pré-entraînés).
> - [NanoGPT](https://github.com/karpathy/nanoGPT) (une autre implémentation minimaliste).
> - [LLM University](https://github.com/DAIR-AI/ML-Notebooks) (pour approfondir).