import random
from typing import Any
import numpy as np
import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fractions import Fraction

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

def create_dict_from_tuples(tuples, complementaire):
    # Create a set of all elements at positions 1 and 2 in the tuples
    elements = set()
    for t in tuples:
        elements.add(t[0])
        elements.add(t[1])

    # Create a dictionary where each key is an element from the set
    # and the value is the complementaire for key 0
    element_dict = {element: complementaire if element == max(elements) else None for element in elements}
    
    return element_dict

    # Find the maximum key
    max_key = max(dico.keys())

    # While there are None values in the dictionary
    while None in dico.values():
        # Iterate over the tuples
        for t in tuples:
            # If the first element of the tuple is the maximum key and the second element is None in the dictionary
            if t[0] == max_key and dico[t[1]] is None:
                # Assign the value to the second element
                dico[t[1]] = chr(ord(dico[max_key]) + t[2])
            # If the second element of the tuple is the maximum key and the first element is None in the dictionary
            elif t[1] == max_key and dico[t[0]] is None:
                # Assign the value to the first element
                dico[t[0]] = chr(ord(dico[max_key]) - t[2])
        # Update the maximum key to the next key that has a value
        max_key = next(key for key in sorted(dico.keys(), reverse=True) if dico[key] is not None)

    return dico

filtered_keys = lambda diction: [key for key, value in diction.items() if value is not None]
 
# def assign_values(dico,tuples, already_done,complement):
#     if len(already_done) == len(dico.keys()):
#         return dico

#     else:
#         todo = [ele for ele in filtered_keys(dico) if ele not in already_done]
#         alred_cp = []
#         for val in todo:
#             for tpl in tuples:
#                 distance = tpl[2]
#                 if val == tpl[0] and tpl[1] not in already_done:
#                     dico_key = tpl[1]
#                     dico[dico_key] = chr(ord(dico[val]) - distance)
#                 elif val == tpl[1] and tpl[0] not in already_done:
#                     dico_key = tpl[0]
#                     dico[dico_key] = chr(ord(dico[val]) + distance)
                
#                 alred_cp = already_done + [val]
                                
#         return assign_values(dico,tuples, alred_cp, complement)


# else of creergraph1
# dataOcc = {}
#     for composant, count in composant_counts.items():
#         if count == 1:
#             graphe.add_node(composant)
#         else:
#             if f"{composant}" in dataOcc.keys():
#                 graphe.add_node(f'{composant}{dataOcc[composant]+1}')
#                 dataOcc[composant] += 1
#             else:
#                 graphe.add_node(f'{composant}1')
#                 dataOcc[composant] = 1
            
#     afficherGraphe(graphe)


saisieUtilisateur = ""

def interfaceGraphique():
    root = tk.Tk()

    # Demander à l'utilisateur de saisir un mot à chiffrer
    label = tk.Label(root, text="Veuillez entrer un mot :")
    label.pack()
    my_entry = tk.Entry(root)
    my_entry.pack()
    bouton = tk.Button(root, text="Valider", command=lambda: valider(my_entry, root))
    bouton.pack()
    root.mainloop()

def valider(my_entry, root):
    global saisieUtilisateur

    # Stocker la saisie de l'utilisateur dans saisieUtilisateur
    saisieUtilisateur = my_entry.get()
    root.destroy() 

# Créer le graphe1, dont chaque sommet est un composant de saisieUtilisateur
def creerGraphe1(saisieUtilisateur): 
    graphe = nx.Graph()
    composant_counts = {}

    for composant in saisieUtilisateur:
        composant_counts[composant] = composant_counts.get(composant, 0) + 1

    for composant, count in composant_counts.items():
        if count == 1:
            graphe.add_node(composant)
        else:
            for i in range(1, count + 1):
                graphe.add_node(f"{composant}{i}")

    return graphe

# version debug
# dataOcc = {}
#     for composant, count in composant_counts.items():
#         if count == 1:
#             graphe.add_node(composant)
#         else:
#             if f"{composant}" in dataOcc.keys():
#                 graphe.add_node(f'{composant}{dataOcc[composant]+1}')
#                 dataOcc[composant] += 1
#             else:
#                 graphe.add_node(f'{composant}1')
#                 dataOcc[composant] = 1
            
# fin

# version normal
# for composant, count in composant_counts.items():
        # if count == 1:
        #     graphe.add_node(composant)
        # else:
        #     for i in range(1, count + 1):
        #         graphe.add_node(f"{composant}{i}")
# fin

# Créer le graphe graphe2, qui est graphe1 avec en plus des arêtes entre chaque composant de saisieUtilisateur
def creerGraphe2(graphe1):
    graphe2 = graphe1.copy()
    composants = list(graphe1.nodes())
    premier_composant, dernier_composant = composants[0], composants[-1]

    if len(composants) > 2 and not graphe2.has_edge(premier_composant, dernier_composant):
        graphe2.add_edge(premier_composant, dernier_composant)

    for i in range(len(composants) - 1):
        composant1, composant2 = composants[i], composants[i + 1]
        if not graphe2.has_edge(composant1, composant2):
            graphe2.add_edge(composant1, composant2)

    return graphe2

# Créer le graphe graphe3, qui est graphe2 avec en plus comme poids les distances entre les composants les sommets selon le code ASCII
def creerGraphe3(graphe2):
    graphe3 = graphe2.copy()
    nodes = list(graphe3.nodes())
    first_node, last_node = nodes[0], nodes[-1]

    for u, v, attr in graphe3.edges(data=True):
        distance_ascii = ord(v[0]) - ord(u[0])
        if (u == first_node and v == last_node) or (u == last_node and v == first_node):
            distance_ascii = -distance_ascii
        attr['weight'] = distance_ascii

    return graphe3

# Créer le graphe graphe4, qui est graphe3 avec en plus des arêtes en plus pour que tous les sommets soient reliés et en leur ajoutant des poids débutant à 129
def creerGraphe4(graphe3):
    graphe4 = graphe3.copy()
    nodes = list(graphe4.nodes())
    poids_initial = 129 

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if not graphe4.has_edge(u, v):
                graphe4.add_edge(u, v, weight=poids_initial)
                poids_initial += 1  

    return graphe4

# Créer graphe 5, qui est graphe4 avec en plus un caractère aléatoire utilisé comme sommet, lié à un sommet aléatoire de graphe4 et dont le poids est la distance entre ce caractère aléatoire et le sommet auquel il est lié
def creerGraphe5(graphe4, caractere_supplementaire, position):
    graphe5 = graphe4.copy()
    nodes = list(graphe4.nodes())
    sommet_aleatoire = nodes[position] if position is not None and 0 <= position < len(nodes) else random.choice(nodes)
    distance_ascii = ord(sommet_aleatoire[0]) - ord(caractere_supplementaire)

    graphe5.add_node(caractere_supplementaire)
    graphe5.add_edge(caractere_supplementaire, sommet_aleatoire, weight=distance_ascii)

    return graphe5

def afficherGraphe(graphe):
    fig, ax = plt.subplots()
    pos = nx.circular_layout(graphe)
    colors = cm.rainbow(np.linspace(0, 1, len(graphe.edges)))
    edge_colors = [colors[i] for i in range(len(graphe.edges))]
    
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

# Créer la matrice de distance X1 pour graphe5
def creerX1(graphe, saisieModifiee, caractere_supplementaire):
    taille = len(saisieModifiee)
    X1 = np.zeros((taille, taille), dtype=int)
    saisieModifieeDict = modifierComposants(saisieModifiee, caractere_supplementaire)
    
    for u, v, attr in graphe.edges(data=True):
        indices_u = [i for i, ch in enumerate(saisieModifiee) if ch == u or saisieModifieeDict[ch] == u]
        indices_v = [i for i, ch in enumerate(saisieModifiee) if ch == v or saisieModifieeDict[ch] == v]
        
        for indice_u in indices_u:
            for indice_v in indices_v:
                poids = attr['weight']
                X1[indice_u, indice_v] = poids
                X1[indice_v, indice_u] = poids

    return X1
    
def Swap(arr, start_index, last_index):
    arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]]

def creerX2(graphe, saisieModifiee):
    # Nombre de sommets dans le graphe
    nb_sommets = len(graphe.nodes())

    # Initialiser une matrice de distances avec des valeurs infinies
    X2 = np.full((nb_sommets, nb_sommets), -10000 )

    # Remplir la matrice avec les distances entre les sommets
    for u, v, data in graphe.edges(data=True):
        poids = round(data['weight'])  # Arrondir le poids à l'entier le plus proche
        indice_u = list(graphe.nodes()).index(u)
        indice_v = list(graphe.nodes()).index(v)
        X2[indice_u, indice_v] = poids
        X2[indice_v, indice_u] = poids  # La matrice est symétrique pour un graphe non orienté

    # Mettre les indices des composants de saisieModifiee sur la diagonale de X2
    for i, composant in enumerate(saisieModifiee):
        indice_composant = list(graphe.nodes()).index(composant)
        X2[indice_composant, indice_composant] = 0  # La distance d'un sommet à lui-même est 0
    # Swap(X2, 0, -1)
    return X2

def estInversible(matrice):
    return np.linalg.matrix_rank(matrice) == len(matrice)

def modifierComposants(saisieModifiee, caractere_supplementaire):
    nouvelle_saisie = {}
    composant_counts = {}

    for composant in saisieModifiee:
        composant_counts[composant] = composant_counts.get(composant, 0) + 1

    for composant, count in composant_counts.items():
        if count == 1:
            nouvelle_saisie[composant] = composant
        else:
            for i in range(1, count + 1):
                if composant != caractere_supplementaire:
                    nouvelle_saisie[f"{composant}{i}"] = composant
                else:
                    if i == 1:
                        nouvelle_saisie[composant] = composant
                    else:
                        nouvelle_saisie[f"{composant}{i}"] = composant

    return nouvelle_saisie

def Kruskal(graphe):
    poidsAretes = {}

    for u, v, attr in graphe.edges(data=True):
        poids = attr['weight']
        arrete = (u, v)
        poidsAretes[poids] = arrete
    myKeys = list(poidsAretes.keys())
    myKeys.sort()
    poidsOrdreCroissant = {i: poidsAretes[i] for i in myKeys}

    grapheKruskal  = nx.Graph()

    nodes = list(graphe.nodes)
    for node in nodes:
        grapheKruskal.add_node(node)

    for p in poidsOrdreCroissant:
        u, v = poidsAretes[p]
        if nx.has_path(grapheKruskal, u, v):  
            continue
        grapheKruskal.add_edge(u, v, weight=p)

    return grapheKruskal

def creerPk(taille):
    while True:
        matrice = np.random.randint(10, size=(taille, taille))

        if np.linalg.matrix_rank(matrice) == taille:
            return matrice

# Créer le graphe graphe7 à partir de X2
def creerDecryptedGraph1(X2):
    graphe7 = nx.Graph()

    # Nombre de sommets dans le graphe
    nb_sommets = len(X2)

    # Ajouter les sommets au graphe
    for i in range(nb_sommets):
        graphe7.add_node(i)

    for i in range(nb_sommets-1):
        # add edge based on number: 1-2, 2-3, 3-4...
        if (i < nb_sommets - 1 and X2[i][i + 1] != -10000 ): #or (i < nb_sommets - 1 and X2[i][0] == 0 and i >=1 ):
            graphe7.add_edge(i, i + 1, weight=X2[i][i + 1])
    # Ajouter les arêtes avec leurs poids
    for i in range(nb_sommets):
        for j in range(nb_sommets):
            poids = X2[i][j]
            if poids != -10000 and j != i+1  and i!= j:
                graphe7.add_edge(i, j, weight=poids)
    
    
    # changer les valeurs 0 1 2 3 ....
    for i in range(nb_sommets):
         graphe7 = nx.relabel_nodes(graphe7, {i: i})

    # je veux changer le des sommets 0 en -1 et -1 en 0
    graphe7 = nx.relabel_nodes(graphe7, {nb_sommets: 100})
    
    edges_with_weights = [(u, v, d['weight']) for u, v, d in graphe7.edges(data=True)]
    return graphe7, edges_with_weights




def retrouverMotChiffre(graphe, X2):
    mot_chiffre = ""
    
    # Récupérer les sommets du graphe et les trier par ordre de valeur
    sommets = sorted(graphe.nodes())
    
    # Parcourir les sommets dans l'ordre trié
    for sommet in sommets:
        # Récupérer les voisins du sommet actuel dans le graphe
        voisins = graphe.neighbors(sommet)
        
        # Parcourir les voisins et ajouter les poids des arêtes correspondantes à mot_chiffre
        for voisin in voisins:
            # Récupérer le poids de l'arête entre le sommet actuel et son voisin
            poids = graphe[sommet][voisin]['weight']
            
            # Ajouter le poids à mot_chiffre
            mot_chiffre += str(poids)
    
    return mot_chiffre



    


def main():
    ### Chiffrer ###

    caractere_supplementaire = 'A'
    #'►'
    position = 0

    interfaceGraphique()
    
    # Créer le graphe1, dont chaque sommet est un composant de saisieUtilisateur
    graphe1 = creerGraphe1(saisieUtilisateur)

    # Créer le graphe graphe2, qui est graphe1 avec en plus des arêtes entre chaque composant de saisieUtilisateur
    graphe2 = creerGraphe2(graphe1)

    # Créer le graphe graphe3, qui est graphe2 avec en plus comme poids les distances entre les composants selon le code ASCII
    graphe3 = creerGraphe3(graphe2)

    # Créer le graphe graphe4, qui est graphe3 avec en plus des arêtes pour que tous les sommets soient reliés et en leur ajoutant des poids débutant à 129
    graphe4 = creerGraphe4(graphe3)
    # Créer graphe 5, qui est graphe4 avec en plus un caractère aléatoire utilisé comme sommet, lié à un sommet aléatoire de graphe4 et dont le poids est la distance entre ce caractère aléatoire et le sommet auquel il est lié
    graphe5 = creerGraphe5(graphe4, caractere_supplementaire, position)
    # Afficher graphe5
    afficherGraphe(graphe5)

    saisieModifiee = saisieUtilisateur[:position] + caractere_supplementaire + saisieUtilisateur[position:]
    saisieModifiee = modifierComposants(saisieModifiee, caractere_supplementaire)

    # Créer la matrice de distance X1 pour graphe5
    X1 = creerX1(graphe5, saisieModifiee, caractere_supplementaire)

    print(estInversible(X1))
    print(X1)

    while (position<len(saisieModifiee) and not estInversible(X1)):
        position += 1
        graphe5 = creerGraphe5(graphe4, caractere_supplementaire, position)
        saisieModifiee = saisieUtilisateur[:position] + caractere_supplementaire + saisieUtilisateur[position:]
        X1 = creerX1(graphe5, saisieModifiee, caractere_supplementaire)  
        print(position)  
        print(estInversible(X1))
        print("X1 :")
        print(X1)

    # Créer le graphe graphe6 en appliquant l'algithme de Kruskal sur graph6
    graphe6 = Kruskal(graphe5)
    afficherGraphe(graphe6)

    # Créer la matrice de distance X2 pour graph6
    X2 = creerX2(graphe6, saisieModifiee)
    print("X2 :")
    print(X2)

    # Calculer X3 = X1 * X2
    X3 = np.dot(X1, X2)
    print("X3 :")
    print(X3)

    taille_matrice = len(graphe6.nodes())

    # Créer une matrice aléatoire Pk inversible de taille nombre de sommets de graph6 * nombre de sommets de graph6
    Pk = creerPk(taille_matrice)
    print("Pk :")
    print(Pk)

    # Calculer Ct = Pk X3
    Ct = np.dot(Pk, X3)
    print("Ct :")
    print(Ct)

    ### Déchiffrer ###
    x3_from_keys = retrouverX(Pk, Ct)
    x2_from_keys = retrouverX(X1, x3_from_keys)
    # Créer le graph_1 à partir de X2 ( toutes les connexions )
    graphe7, edges_with_weights  = creerDecryptedGraph1(x2_from_keys)
    afficherGraphe(graphe7)
    # max_tuple = max(edges_with_weights, key=lambda x: x[1])
    # other_tuples = list(filter(lambda x: x[1] != max_tuple[1], edges_with_weights))
    dico = create_dict_from_tuples(edges_with_weights, caractere_supplementaire)
    print("Dico init:",dico)
    print("Chemins",edges_with_weights)
    # print(assign_values(dico, edges_with_weights,[],caractere_supplementaire))
    #------------------


      
    
    plt.show()


if __name__ == "__main__":
    main()