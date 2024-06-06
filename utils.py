from operator import le
from random import randint
from networkx import Graph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fractions import Fraction
from typing import Any


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
    for lettre in userInput:
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