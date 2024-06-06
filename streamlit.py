from operator import le
from random import randint
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fractions import Fraction
from typing import Any
import streamlit as st

from utils import *

# ----------------- Streamlit -----------------#

def streamlit_main():
    streamlit_process()


def display_matrix(matrix1, matrix2, matrix1_title, matrix2_title):
    col1, col2 = st.columns(2)

    with col1:
        st.write(matrix1_title, unsafe_allow_html=True)
        st.write(matrix1)
    with col2:
        st.write(matrix2_title, unsafe_allow_html=True)
        st.write(matrix2)
    

def stream_data(string_input,stream_time):
    for i in string_input:
        yield i
        time.sleep(stream_time)


def display_graphs(graph1, graph2, graph1_title, graph2_title):
    # Create a two-column layout
    col1, col2 = st.columns(2)

    # Display Graph 6 in the first column
    with col1:
        st.write(graph1_title, unsafe_allow_html=True)
        st.pyplot(graph1, clear_figure=True)
        st.write(" ")  # Add some spacing

    # Display Graph 5 in the second column
    with col2:
        st.write(graph2_title, unsafe_allow_html=True)
        st.pyplot(graph2, use_container_width=True)
        st.write(" ")  # Add some spacing

def process_with_double(the_input):
    double_graphe = nx.Graph()
    double_entreeModifiee = modifierEntree(double_graphe, the_input + the_input[0])
    double_graphe.add_node(caractere_supplementaire)
    double_graphe.add_edge(caractere_supplementaire, double_entreeModifiee[0], weight=ord(double_entreeModifiee[0][0]) - ord(caractere_supplementaire))
    st_graph_fig = afficherGraphe(double_graphe)
    double_entreeModifiee = [caractere_supplementaire] + double_entreeModifiee
    double_x1 = creerX(double_graphe, double_entreeModifiee)
    double_kruskal = Kruskal(double_graphe)
    double_x2 = creerX(double_kruskal, double_entreeModifiee)
    for i in range(len(double_x2)):
        double_x2[i, i] = i
    double_x3 = np.dot(double_x1, double_x2)
    double_pk = creerPk(len(double_entreeModifiee))
    double_ct = np.dot(double_pk, double_x3)
    st_double_graph7 = creerGrapheAPartirMatrice(double_x2)
    st_dictionnaire_final = dechiffrerAvecGraphe(st_double_graph7,caractere_supplementaire)
    
    st.write(st_dictionnaire_final)        
    st.write_stream(stream_data(f"""Explication de l'algorithme""",0.02))
    # Etape 6
    st.write_stream(stream_data(f"""### Etape 5: Reconstitution du message""",0.02))
    mot_trouve = assembler_message(st_dictionnaire_final)
    st.write(f"Le message trouvé est : ")
    st.write(f"`{mot_trouve}`")
    st.toast('Méssage déchiffré avec succès', icon="🔓")

    # Début des remarques

    st.markdown(f"""## <ins>Remarque:</ins>""",unsafe_allow_html=True)
    
    st.error(f"""Note: \n > Pour pouvoir appliquer l'algorithme de Kruskal, nous avons utilisé un graphe non orienté. Or, pour reconstituer correctement le message, nous avons besoin de connaître le sens des arêtes. Cela nous permettra de savoir comment les poids avaient été calculés au départ. Ainsi, nous avons ajouté le premier caractère de la phrase saisie à la fin de la phrase. Il est inconnu des tiers dans tous les cas. Donc il ne compromet pas le chiffrement.""")
    
    st.write_stream(stream_data(f"""###### Nouveau graphe complet""",0.02))
    st.pyplot(st_graph_fig)

    st.write_stream(stream_data(f"""###### Matrice de distance X1""",0.02))

    st.write(double_x1)
    
    display_graphs(st_graph_fig, afficherGraphe(double_kruskal), f""" ###### Graphe complet""", f"""###### Kruskal""")

    st.write_stream(stream_data(f"""###### Matrice de distance X2""",0.02))

    st.write(double_x2)

    st.write_stream(stream_data(f"""###### Matrice de distance X3""",0.02))
    st.write(double_x3)

    st.write_stream(stream_data(f"""###### Matrice aléatoire Pk""",0.02))
    st.write(double_pk)

    st.write_stream(stream_data(f"""###### Calcul de Ct""",0.02))
    st.write(double_ct)

    
    st.write_stream(stream_data(f"""###### Elements envoyés""",0.02))
    display_matrix(double_x1, double_ct, f"""##### X1""", f"""##### Ct""")
    
    # Déchiffrement
    st.markdown(f"""## <ins>Déchiffrement:</ins>""",unsafe_allow_html=True)
   
    # Etape 1
    st.write_stream(stream_data(f"""### Etape 1: Récupération de X3 à partir de Ct""",0.02))
    st.latex(r"X3 = Pk^{-1} \times Ct")
    st_double_X3_from_keys = retrouverX(double_pk, double_ct)
    st.write(st_double_X3_from_keys)
    st.toast('Matrice X3 recalculé avec succès', icon="🔓")


    # Etape 2
    st.write_stream(stream_data(f"""### Etape 2: Récupération de X2 à partir de X3 et X1""",0.02))
    st.latex(r"X2 = X1^{-1} \times X3")
    double_x2_from_keys = retrouverX(double_x1, st_double_X3_from_keys)
    st.toast('Matrice X2 recalculé avec succès', icon="🔓")
    st.write(double_x2_from_keys)
    
    # Etape 3
    st.write_stream(stream_data(f"""### Etape 3: Création du nouveau graph en fonction de X2""",0.02))
    double_st_graph7 = creerGrapheAPartirMatrice(double_x2_from_keys)
    double_st_graph7_fig = afficherGraphe(double_st_graph7)
    st.pyplot(double_st_graph7_fig)
    st.write_stream(stream_data(f""" On peut remarquer que ce graphe a la _**même forme que le graphe de l'arbre couvrant minimal**_. Toutefois, celui-ci n'a pas de lettre comme nom de sommet mais des numéros qui représentent l'ordre des lettres.""",0.02))

     # Etaoe 4
    st.write_stream(stream_data(f"""### Etape 4: Reconstruire le message""",0.02))
    st.write_stream(stream_data(f"""##### Reconstitution du dictionnaire""",0.02))
    st.write(st_dictionnaire_final)        
    st.write_stream(stream_data(f"""Explication de l'algorithme""",0.02))
    # Etape 5
    st.write_stream(stream_data(f"""### Etape 5: Reconstitution du message""",0.02))
    mot_trouve = assembler_message(st_dictionnaire_final)
    st.write(f"Le message trouvé est : ")
    st.write(f"`{mot_trouve}`")
    st.toast('Méssage déchiffré avec succès', icon="🔓")


def streamlit_process():

    count = 0
    st.title("Suivez le processus de chiffrement et de déchiffrement de votre message")
    userInput = st.chat_input("Say something")
    if userInput:
        st.toast('Message enregistré avec succès', icon="📝")
        count += 1
        if count > 1:
            st.toast('Veuillez scroller en haut de la page pour voir le processus depuis le début', icon="🔄")
        
        # run the scroll code 
        st.markdown(f"""## <ins>Chiffrement:</ins>""",unsafe_allow_html=True)
        st.write_stream(stream_data(f"""Le message à chiffrer est:""",0.02))
        st.code(userInput)
        

        # Etape 0: initialisation 
        graphe = nx.Graph()
        entreeModifiee = modifierEntree(graphe, userInput)
        graphe.add_node(caractere_supplementaire)
        graphe.add_edge(caractere_supplementaire, entreeModifiee[0], weight=ord(entreeModifiee[0][0]) - ord(caractere_supplementaire))
 
        # Etape 1
        st.write_stream(stream_data(f"""### Etape 1: Réalisation du graph complet""",0.02))
        st_graph5_fig = afficherGraphe(graphe)
        st.pyplot(st_graph5_fig)
        st.write_stream(stream_data(f"""On a d'abord créé un cycle reliant tous les composants de la saisie de l'utilisateur 
Puis nous avons ajouté des arêtes ainsi que la distance entre les deux sommet. Toutefois, le graphe utilisé n'est pas orienté.""",0.02))
        
        st.warning(f"""Note: \n > Afin d'éviter d'avoir un graphe non connexes après avoir appliqué Kruskal, nous avons initialisé la distance entre deux sommets identiques à 0 et deux sommet non reliés à -300.
Ça nous permettra de distinguer les arêtes inexistantes des arêtes entre deux fois le même caractère.  -300 est une valeur tellement petite qu'elle ne transformera probablement pas une valeur non nulle du tableau en 0. C'est vrai pour une phrase de plus de 171 caractères (300-128)
Effectivement 128, c'est le premier poids qu'on ajoute entre des lettres qui ne se suivent pas. Puis les poids entre les caractères qui ne se suivent pas augmentent de manière incrémentale à partir de 128""")

        # Etape 2
        st.write_stream(stream_data(f"""### Etape 2: Création de la matrice de distance X1""",0.02))
        st.write_stream(stream_data("_**X1**_ est la matrice de distance pour la saisie utilisateur",0.02))
        entreeModifiee = [caractere_supplementaire] + entreeModifiee
        X1 = creerX(graphe, entreeModifiee)
        st.write(X1)
        st.write_stream(stream_data(f"""La _**matrice X1 { "est inversible" if matriceInversibleOuNon(X1) == 1 else "n'est pas inversible"}**_""",0.02))
        
        
        # Etape 3
        st.write_stream(stream_data(f"""### Etape 3: Réalisation du graphe minimal""",0.02))
        st.write_stream(stream_data(f"""##### Application de l'algorithme de Kruskal""",0.02))
        st_graph6 = Kruskal(graphe)
        st_graph6_fig = afficherGraphe(st_graph6)
        display_graphs(st_graph5_fig, st_graph6_fig, f""" ###### Graphe complet""", f"""###### Kruskal""")

        # Etape 4
        st.write_stream(stream_data(f"""### Etape 4: Création de la matrice du graph de Kruskal""",0.02))
        st.caption("_**X2**_ est la matrice de distance du graphe couvrant minimal ")
        X2 = creerX(st_graph6, entreeModifiee)
        for i in range(len(X2)):
            X2[i, i] = i  
        st.write(X2)

        # Etape 5
        st.write_stream(stream_data(f"""### Etape 5: Création de la matrice X3""",0.02))
        st.latex(r"X3 = X1 \times X2")
        X3 = np.dot(X1, X2)
        st.write(X3)

        # Etape 6
        st.write_stream(stream_data(f"""### Etape 6: Création de la matrice aléatoire Pk""",0.02))
        st.caption("Pk est une matrice aléatoire unimodulaire dont la taille est égale à (longueur de la phrase saisie + 1) * (longueur de la phrase saisie + 1).")
        st_Pk =  creerPk(len(entreeModifiee))
        st.write(st_Pk)

        # Etape 7
        st.write_stream(stream_data(f"""### Etape 7: Calcul de Ct""",0.02))
        st.latex(r"Ct = Pk \times X3")
        st_Ct = np.dot(st_Pk, X3)
        st.write(st_Ct)
        st.write_stream(stream_data(f"""_**Ct**_ représente la matrice du message chiffré""",0.02))
        st.write_stream(stream_data(f"""La valeur qui sera par conséquent envoyée au recepteur est: (X1, Ct).""",0.02))
        st.toast('Méssage chiffré avec succès', icon="🔒")
        st.write(f"""<ins>Elements envoyé</ins>: """)
        display_matrix(X1, st_Ct, f"""##### X1""", f"""##### Ct""")
        
        
        # Déchiffrement
        st.markdown(f"""## <ins>Déchiffrement:</ins>""",unsafe_allow_html=True)
        # Etape 1
        st.write_stream(stream_data(f"""### Etape 1: Récupération de X3 à partir de Ct""",0.02))
        st.latex(r"X3 = Pk^{-1} \times Ct")
        X3_from_keys = retrouverX(st_Pk, st_Ct)
        st.write(X3_from_keys)
        st.toast('Matrice X3 recalculé avec succès', icon="🔓")

        # Etape 2
        st.write_stream(stream_data(f"""### Etape 2: Récupération de X2 à partir de X3 et X1""",0.02))
        st.latex(r"X2 = X1^{-1} \times X3")
        X2_from_keys = retrouverX(X1, X3_from_keys)
        st.toast('Matrice X2 recalculé avec succès', icon="🔓")
        st.write(X2_from_keys)
       
        # Etape 3
        st.write_stream(stream_data(f"""### Etape 3: Création du nouveau graph en fonction de X2""",0.02))
        st_graph7 = creerGrapheAPartirMatrice(X2_from_keys)
        st_graph7_fig = afficherGraphe(st_graph7)
        st.pyplot(st_graph7_fig)
        st.write_stream(stream_data(f""" On peut remarquer que ce graph a la _**même forme que l'arbre couvrant minimale**_. Toutefois, on peut remarquer que _**celui-ci n'a pas de lettre comme nom de sommet mais des numéros**_ qui _**représentent l'ordre des lettres**_""",0.02))
        
        # Etaoe 4
        st.write_stream(stream_data(f"""### Etape 4: Reconstruire le message""",0.02))
        st.write_stream(stream_data(f"""##### Reconstitution du dictionnaire""",0.02))
        
        process_with_double(userInput)

# ----------------- Streamlit -----------------#
if __name__ == "__main__":
    caractere_supplementaire = '╩'
    streamlit_main()