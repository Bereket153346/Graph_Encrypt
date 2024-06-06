# Chiffrement et Déchiffrement avec Kruskal

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0-red)
![Tkinter](https://img.shields.io/badge/Tkinter-UI-blueviolet)
![VS Code](https://img.shields.io/badge/VS%20Code-Editor-blue)

### Projet d'Optimisation et Complexité

---

## Introduction

Dans ce projet, nous implémentons un algorithme de chiffrement et de déchiffrement en utilisant la théorie de graphes, plus précisément le concept d’arbre couvrant minimal. Cet algorithme est symétrique, ce qui signifie que la même clé privée est utilisée pour chiffrer et déchiffrer le message.

![image](https://github.com/ilhem033/projet/assets/171623069/bd3e66eb-aa3c-492a-a9ed-a45c25e9eaff)

## Algo de Kruskal

Ici, nous avons choisi d’utiliser l’algorithme de Kruskal afin de trouver l’arbre couvrant minimal. L’algorithme de Kruskal est une méthode pour trouver l’arbre couvrant minimal d’un graphe pondéré, non orienté. Il commence par trier toutes les arêtes du graphe en ordre croissant de poids. Puis, il ajoute les arêtes une par une à l’arbre couvrant, en évitant de créer des cycles, jusqu’à ce que tous les nœuds soient connectés. Cet algo est efficace puisqu’il garantit la minimisation du coût total des arêtes dans l’arbre couvrant, ce qui est essentiel dans des applications comme le chiffrement basé sur la théorie des graphes.

Ce rapport a pour objectif d’expliquer le fonctionnement de notre implémentation en Python. La raison pour laquelle nous utilisons un arbre couvrant de poids minimal et pourquoi l’usage de la matrice X1 seule ne suffit pas pour déchiffrer le message.

## Implémentation du code Python

Pour concevoir ce projet, nous avons utilisé plusieurs modules Python.

![Code Implementation Image](path_to_image)

## Utilisation de Streamlit

Dans ce projet, nous utilisons Streamlit pour créer une interface utilisateur conviviale permettant d'afficher les matrices et les graphes générés lors du processus de chiffrement et de déchiffrement.

### Affichage des Matrices

Streamlit nous permet d'afficher les matrices générées par notre algorithme de chiffrement de manière claire et organisée. Nous utilisons des composants Streamlit tels que `st.dataframe()` pour afficher les matrices sous forme de tableaux faciles à lire.

### Affichage des Graphes

Pour visualiser les graphes générés lors de l'application de l'algorithme de Kruskal pour trouver l'arbre couvrant minimal, nous utilisons les fonctionnalités de visualisation de Streamlit. Avec des modules Python comme Matplotlib ou Plotly, nous générons les graphes correspondant aux données de manière interactive, permettant ainsi aux utilisateurs d'explorer les résultats plus en profondeur.

Streamlit simplifie grandement le processus de création d'une interface utilisateur interactive pour notre projet, ce qui le rend plus accessible et convivial pour les utilisateurs finaux.

## Conclusion

Ce projet démontre l’application pratique de la théorie des graphes dans la cryptographie. L’utilisation d’un arbre couvrant minimal optimise le processus de chiffrement en minimisant les distances et les coûts associés. En combinant cela avec des matrices aléatoires pour le chiffrement, nous assurons la sécurité des données transmises. Bien que la matrice X1 soit publique, elle ne suffit pas à elle seule pour déchiffrer le message sans la clé privée Pk.

## Instructions d'installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/lionelmarcus10/graph_encrypt.git
   ```
2. Installation de Steamlit :
   ```bash
   pip install streamlit
   ```
3. Exécution de l'application Steamlit :
   ```bash
   streamlit run main.py
   ```

## Auteurs

- Lionel
- Bereket
- Yilizire
- Steve
- Ilhem
- Marin

L3-APP RS1
