from operator import le
from random import randint
import time
from networkx import Graph
import numpy as np
import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fractions import Fraction
from typing import Any
import streamlit as st
userInput = ""

def interfaceGraphique():
    root = tk.Tk()
    label = tk.Label(root, text="Veuillez entrer un mot :")
    label.pack()
    my_entry = tk.Entry(root)
    my_entry.pack()
    bouton = tk.Button(root, text="Valider", command=lambda: valider(my_entry, root))
    bouton.pack()
    root.mainloop()

def valider(my_entry, root):
    global userInput
    userInput = my_entry.get()
    root.destroy() 

def afficherGraphe(graphe):
    fig, ax = plt.subplots()
    pos = nx.circular_layout(graphe)

    # Créer une palette de couleurs
    colors = cm.rainbow(np.linspace(0, 1, len(graphe.edges)))
    edge_colors = [colors[i] for i in range(len(graphe.edges))]
    
    #nx.draw(graphe, pos, ax=ax, with_labels=True)
    nx.draw(graphe, pos, ax=ax, with_labels=True, edge_color=edge_colors)

    edge_pos = nx.circular_layout(graphe)

    edge_labels = nx.get_edge_attributes(graphe, 'weight')

    offset = 0.05

    
    for i, ((u, v), weight) in enumerate(edge_labels.items()):
        x1, y1 = edge_pos[u]
        x2, y2 = edge_pos[v]
        label_pos_x = (x1 + x2) / 2 + offset * (y2 - y1)
        label_pos_y = (y1 + y2) / 2 - offset * (x2 - x1)
        plt.text(label_pos_x, label_pos_y, weight, horizontalalignment='center', verticalalignment='center', color=edge_colors[i])

    return fig
    # plt.show()

def creerX(graphe, entreeModifiee):
    taille = len(entreeModifiee)
    X1 = np.full((taille, taille), -300, dtype=int)
    for u, v, attr in graphe.edges(data=True):
        i = entreeModifiee.index(u)
        j = entreeModifiee.index(v)
        poids = attr['weight']
        X1[i, j] = poids
        X1[j, i] = poids 
    return X1

def creerPk(taille):
    while True:
        matrice = np.random.randint(10, size=(taille, taille))

        if np.linalg.matrix_rank(matrice) == taille:
            return matrice

def eliminate(r1, r2, col, target=0):
    fac = Fraction((r2[col]-target), r1[col])
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]

def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(i+1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                raise ValueError("Matrix is not invertible")
        for j in range(i+1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a)-1, -1, -1):
        for j in range(i-1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a

def inverse(a):
    tmp = [[] for _ in a]
    for i,row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))
    gauss(tmp)
    return [tmp[i][len(tmp[i])//2:] for i in range(len(tmp))]

def retrouverX(x1: np.ndarray[Any],x3):
    # x2 = x1^-1 * x3
    x1_simple = x1.tolist()
    x1_inversed = inverse(x1_simple)
    x1_np_array_renewed = np.array(x1_inversed)
    new_x2 = np.dot(x1_np_array_renewed, x3)
    try:
        numerators = np.vectorize(lambda x: x.numerator)(new_x2)
        # Step 3: Convert the array of numerators to a regular NumPy array with integers
        integer_array = numerators.astype(int)
        return integer_array
    except:
        return new_x2

def Kruskal(graphe):
    # Créer une liste des arêtes avec leur poids
    aretes_ponderees = [(u, v, attr['weight']) for u, v, attr in graphe.edges(data=True)]
    # Trier les arêtes par poids
    aretes_ponderees.sort(key=lambda x: x[2])
    
    # Créer un nouvel graphe pour l'arbre couvrant minimal
    graphe_arbre_couvrant = nx.Graph()
    
    # Ajouter tous les sommets du graphe original
    graphe_arbre_couvrant.add_nodes_from(graphe.nodes)
    
    # Parcourir toutes les arêtes triées
    for u, v, poids in aretes_ponderees:
        # Vérifier si l'ajout de l'arête crée un cycle dans le graphe de l'arbre couvrant
        if not nx.has_path(graphe_arbre_couvrant, u, v):
            # Ajouter l'arête dans le graphe de l'arbre couvrant
            graphe_arbre_couvrant.add_edge(u, v, weight=poids)
        
        # Arrêter la recherche dès que tous les sommets sont connectés
        if nx.number_of_edges(graphe_arbre_couvrant) == (nx.number_of_nodes(graphe) - 1):
            break
    
    return graphe_arbre_couvrant


def creerGrapheAPartirMatrice(matrice):
    graphe = nx.Graph()
    for i in range(len(matrice)):
        for j in range(len(matrice)):
            if i != j and matrice[i][j] != -300:  
                graphe.add_edge(i, j, weight=matrice[i][j])

    return graphe

def matriceInversibleOuNon(matrice):
    try:
        np.linalg.inv(matrice)
        return 1
    except np.linalg.LinAlgError:
        return 0
    
def dechiffrerAvecGraphe(graphe: Graph, caractere_supplementaire):
    # print(edges_with_weights)
    # print(graphe.nodes(data=False))
    # print(graphe.get_edge_data(5,4))

    # initialiser le dictionnaire pour les lettres du message final avec None
    message_final =  {u:  None if u != 0 else caractere_supplementaire for u in graphe.nodes(data=False)}
    already_done = [0]

    # parcourir le dictionnaire tant que tous les sommets n'ont pas de lettre
    while None in message_final.values():
        # parcourir tous les noeuds
        for u in graphe.nodes(data=False):
            # verifier si le noeud a deja ete attribue une lettre
              if u in already_done:
                  # chercher tous les arrets connectés à ce noeud
                  for i in graphe.neighbors(u):
                    # verifier si le noeud a deja ete attribué une lettre
                    if message_final[u] != None and message_final[i] == None:
                        valeur_arret_avec_direction_inconnu = graphe.get_edge_data(u,i)['weight']
                        valeur_arret_direction_connu = valeur_arret_avec_direction_pour_graph_sans_direction(u,i, valeur_arret_avec_direction_inconnu)
                        final_letter = ascii_operation(message_final[u], valeur_arret_direction_connu )
                        message_final[i] = final_letter
                        # attribuer la lettre en fonction des distances
                        already_done.append(i)
    return message_final




def retrouver_ordre_message(graph: Graph) -> int:
    message_final =  {u:  None if u != 0 else 0 for u in graph.nodes(data=False)}
    already_done = [0]
    while None in message_final.values():
        # parcourir tous les noeuds
        for u in graph.nodes(data=False):
            # verifier si le noeud a deja ete attribue une lettre
              if u in already_done:
                  # chercher tous les arrets connectés à ce noeud
                  for i in graph.neighbors(u):
                    # verifier si le noeud a deja ete attribué une lettre
                    if message_final[u] != None and message_final[i] == None:
                        message_final[i] = 0
                        already_done.append(i)
    return already_done

def indice_avant(liste: list, a, b):
    index_a = liste.index(a)
    index_b = liste.index(b)
    # On retourne True si l'index de B est avant celui de A
    return index_b < index_a

def valeur_arret_avec_direction_pour_graph_sans_direction(u, v, val):
    end = val
    if u > v:
        end = -val
    return end

def ascii_operation(character, offset):
    # convertir le caractère en valeur ASCII
    ascii_value = ord(character)
    
    # Ajouter le décalage à la valeur ASCII
    new_ascii_value = ascii_value + offset
    
    # Assurer que la nouvelle valeur ASCII reste dans la plage 0-127
    new_ascii_value = new_ascii_value % 128
    
    # Convertir la nouvelle valeur ASCII en caractère
    new_character = chr(new_ascii_value)
    
    return new_character

def assembler_message(dico):
    
    # On trie le dictionnaire par ordre de clé
    dico_trie = sorted(dico.items(), key=lambda item: item[0])
    message = ""
    # On itère sur les paires clé-valeur du dictionnaire trié
    for _, valeur in dico_trie:
        # On retourne chaque caractère de la valeur
        for caractere in valeur:
           message += caractere

    return message[1:-1]

def modifierEntree(graphe: Graph, userInput: str):
    occurrences = {}
    entry_modif = []
    for lettre in userInput + userInput[0]:
        if lettre in occurrences:
            occurrences[lettre] += 1
            entry_modif.append(f"{lettre}{occurrences[lettre]}")
        else:
            occurrences[lettre] = 1
            entry_modif.append(lettre)
        
    for i in range(len(entry_modif)):
        graphe.add_edge(entry_modif[i], entry_modif[i-1], weight=ord(entry_modif[i][0]) - ord(entry_modif[i-1][0]))

    poidsAdditionnel = 127
    for lettre1 in entry_modif:
        for lettre2 in entry_modif:
            if lettre1 != lettre2 and not graphe.has_edge(lettre1, lettre2):
                graphe.add_edge(lettre1, lettre2, weight=poidsAdditionnel)
                poidsAdditionnel += 1
    return entry_modif


def main(caractere_supplementaire):
    interfaceGraphique()

    graphe = nx.Graph()

    entreeModifiee = modifierEntree(graphe, userInput)
    graphe.add_node(caractere_supplementaire)
    graphe.add_edge(caractere_supplementaire, entreeModifiee[0], weight=ord(entreeModifiee[0][0]) - ord(caractere_supplementaire))

    afficherGraphe(graphe)

    entreeModifiee = [caractere_supplementaire] + entreeModifiee
    X1 = creerX(graphe, entreeModifiee)
    print("X1")
    print(X1)

    grapheKruskal = Kruskal(graphe)
    afficherGraphe(grapheKruskal)

    X2 = creerX(grapheKruskal, entreeModifiee)

    for i in range(len(X2)):
        lettre = entreeModifiee[i]
        X2[i, i] = i

    print("X2")
    print(X2)

    # OK
    X3 = np.dot(X1, X2)
    print("X3")
    print(X3)
    
    Pk = creerPk(len(entreeModifiee))
    print("Pk")
    print(Pk)

    Ct = np.dot(Pk, X3)
    print("Ct")
    print(Ct)

    ############### déchiffrer ###############
    X3calcule=retrouverX(Pk, Ct)
    print("X3 calcule")
    print(X3calcule)
    
    X2calcule=retrouverX(X1, X3calcule)
    print("X2 calcule")
    print(X2calcule)

    grapheDechiffrement = creerGrapheAPartirMatrice(X2calcule)
    afficherGraphe(grapheDechiffrement)

    dictionnaire_final = dechiffrerAvecGraphe(grapheDechiffrement,caractere_supplementaire)
    print("message final:", dictionnaire_final)

    mot_trouve = assembler_message(dictionnaire_final)
    print("Le mot trouvé est :", mot_trouve)

    plt.show()

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
        st.write_stream(stream_data(f"""Explication texte with stream""",0.02))
        
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
        st.caption("_**X2**_ est la matrice de distance pour le graphe minimal ( graph de Kruskal )")
        X2 = creerX(st_graph6, entreeModifiee)
        for i in range(len(X2)):
            X2[i, i] = i  
        st.write(X2)
        st.write_stream(stream_data(f"""Expliquer l'utilisation de -300""",0.02))

        # Etape 5
        st.write_stream(stream_data(f"""### Etape 5: Création de la matrice X3""",0.02))
        st.latex(r"X3 = X1 \times X2")
        X3 = np.dot(X1, X2)
        st.write(X3)

        # Etape 6
        st.write_stream(stream_data(f"""### Etape 6: Création de la matrice aléatoire Pk""",0.02))
        st.caption("Pk est une matrice aléatoire unimodulaire dont la taille est égale à la longeur des noeuds du graphe minimal")
        st_Pk =  creerPk(len(entreeModifiee))
        st.write(st_Pk)

        # Etape 7
        st.write_stream(stream_data(f"""### Etape 7: Calcul de Ct""",0.02))
        st.latex(r"Ct = Pk \times X3")
        st_Ct = np.dot(st_Pk, X3)
        st.write(st_Ct)
        st.write_stream(stream_data(f"""_**Ct**_ représente la matrice du message chiffré""",0.02))
        st.write_stream(stream_data(f"""La valeur qui sera oar conséquent envoyé au recepteur est: **(X1, Ct)**""",0.02))
        st.toast('Méssage chiffré avec succès', icon="🔒")
        st.write_stream(stream_data(f"""Elements envoyé: """,0.02))
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
        st.write(X2_from_keys == X2)
       
        # Etape 3
        st.write_stream(stream_data(f"""### Etape 3: Création du nouveau graph en fonction de X2""",0.02))
        st_graph7 = creerGrapheAPartirMatrice(X2_from_keys)
        st_graph7_fig = afficherGraphe(st_graph7)
        st.pyplot(st_graph7_fig)
        st.write_stream(stream_data(f""" On peut remarquer que ce graph a la _**même forme que le graph de Kruskal**_. Toutefois, on peut remarquer que _**celui-ci n'a pas de lettre comme nom de sommet mais des numéros**_ qui _**représentent l'ordre des lettres**_""",0.02))
        
        # Etaoe 4
        st.write_stream(stream_data(f"""### Etape 4: Reconstruire le message""",0.02))
        st.write_stream(stream_data(f"""##### Reconstitution du dictionnaire""",0.02))
        st_dictionnaire_final = dechiffrerAvecGraphe(st_graph7,caractere_supplementaire)
        st.write(st_dictionnaire_final)        
        st.write_stream(stream_data(f"""Explication de l'algorithme""",0.02))
        # Etape 6
        st.write_stream(stream_data(f"""### Etape 5: Dictionnaire final et reconstitution du méssage""",0.02))
        mot_trouve = assembler_message(st_dictionnaire_final)
        st.write(f"Le méssage trouvé est: {mot_trouve}")
        st.toast('Méssage déchiffré avec succès', icon="🔓")


# ----------------- Streamlit -----------------#
if __name__ == "__main__":
    caractere_supplementaire = '╩'

    #main(caractere_supplementaire)
    streamlit_main()