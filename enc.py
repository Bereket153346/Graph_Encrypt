import random
from typing import Any
import numpy as np
import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fractions import Fraction
import streamlit as st
import time
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

filtered_keys = lambda diction: [key for key, value in diction.items() if value is not None]
 
def ascii_operation(character, offset):
    # Convert the character to its ASCII value
    ascii_value = ord(character)
    
    # Perform the operation
    new_ascii_value = ascii_value + offset
    
    # Ensure the result is within the valid ASCII range (0-127)
    new_ascii_value = new_ascii_value % 128
    
    # Convert the ASCII value back to a character
    new_character = chr(new_ascii_value)
    
    return new_character

def assign_values(dico,tuples, already_done):
    if len(already_done) == len(dico.keys()) and all([ele in already_done for ele in dico.keys()]):
        return dico, already_done

    else:
        todo = [ele for ele in filtered_keys(dico) if ele not in already_done]
        print("todooo",todo)
        alred_cp = []
        for val in todo:
            # delimiter la liste des tuples de chemin sur lequel on peut travailler en fonction de ceux qui sont directement connectés
            working_tuples = [tpl for tpl in tuples if val in tpl[:2]]
            for tpl in working_tuples:
                distance = tpl[2]
                if val == tpl[0] and tpl[1] not in already_done and dico[tpl[1]] == None:
                    dico_key = tpl[1]
                    dico[dico_key] = ascii_operation(dico[val],-int(distance))
                    print(f"{tpl} ---=> = {dico_key} : {dico[val]} - {distance} ={ascii_operation(dico[val],-int(distance))} ")

                elif val == tpl[1] and tpl[0] not in already_done and dico[tpl[0]] == None:
                    dico_key = tpl[0]
                    if distance > 0:
                        print("distance positive", distance)
                        dico[dico_key] = ascii_operation(dico[val],int(distance))
                    else:
                        print("distance negative", distance)
                        dico[dico_key] = ascii_operation(dico[val],-int(distance)) # chr( ord(distance))
                    print(f"{tpl} +++++ => = {dico_key} : {dico[val]} + {distance} ={ascii_operation(dico[val],-int(distance))} ")
                else:
                    print(distance)
                alred_cp = already_done + [val]
                                
        return dico, alred_cp



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
    return fig
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

def creerX(graphe, entreeModifiee):
    taille = len(entreeModifiee)
    X1 = np.zeros((taille, taille), dtype=int)
    for u, v, attr in graphe.edges(data=True):
        i = entreeModifiee.index(u)
        j = entreeModifiee.index(v)
        poids = attr['weight']
        X1[i, j] = poids
        X1[j, i] = poids 
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
    nouvelle_saisie = []
    composant_counts = {}

    for composant in saisieModifiee:
        composant_counts[composant] = composant_counts.get(composant, 0) + 1

    for composant, count in composant_counts.items():
        if count == 1:
            nouvelle_saisie.append((composant, composant))
            # nouvelle_saisie[composant] = composant
        else:
            for i in range(1, count + 1):
                if composant != caractere_supplementaire:
                    nouvelle_saisie.append((f"{composant}{i}", composant))
                    # nouvelle_saisie[f"{composant}{i}"] = composant
                else:
                    if i == 1:
                        nouvelle_saisie.append((composant, composant))
                        # nouvelle_saisie[composant] = composant

                    else:
                        nouvelle_saisie.append((f"{composant}{i}", composant))
                        # nouvelle_saisie[f"{composant}{i}"] = composant

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

#  ------------------------------------------------------------ debut streamlit functions ------------------------------------------------------------ #

js_scroll = '''
<script>
    var body = window.parent.document.querySelector(".main");
    console.log(body);
    body.scrollTop = 0;
</script>
'''

def stream_data(string_input,stream_time):
    for i in string_input:
        yield i
        time.sleep(stream_time)


def streamlit_process():

    count = 0
    st.title("Suivez le processus de chiffrement et de déchiffrement de votre message")
    saisieUtilisateur = st.chat_input("Say something")
    if saisieUtilisateur:
        st.toast('Message enregistré avec succès', icon="📝")
        count += 1
        if count > 1:
            st.toast('Veuillez scroller en haut de la page pour voir le processus depuis le début', icon="🔄")
        # run the scroll code 
        saisieModifiee = caractere_supplementaire + saisieUtilisateur
        st.markdown(f"""## <ins>Chiffrement:</ins>""",unsafe_allow_html=True)
        st.write_stream(stream_data(f"""Le message à chiffrer est:""",0.02))
        st.code(saisieUtilisateur)
        
        # Etape 1
        st.write_stream(stream_data(f"""### Etape 1: Réalisation du graph complet""",0.02))
        st_graph5, st_graph5_fig = get_graph_n(saisieUtilisateur, 5)
        st.pyplot(st_graph5_fig)
        st.write_stream(stream_data(f"""Explication texte with stream""",0.02))
        
        # Etape 2
        st.write_stream(stream_data(f"""### Etape 2: Création de la matrice de distance X1""",0.02))
        st.write_stream(stream_data("_**X1**_ est la matrice de distance pour la saisie utilisateur",0.02))
        X1 = creerInversibleX1(position,saisieModifiee,st_graph5)
        st.write(X1)
        st.write_stream(stream_data(f"""La _**matrice X1 { "est inversible" if estInversible(X1) else "n'est pas inversible"}**_""",0.02))
        
        # Etape 3
        st.write_stream(stream_data(f"""### Etape 3: Réalisation du graphe minimal""",0.02))
        st.write_stream(stream_data(f"""##### Application de l'algorithme de Kruskal""",0.02))
        st_graph6 = Kruskal(st_graph5)
        st_graph6_fig = afficherGraphe(st_graph6)
        display_graphs(st_graph5_fig, st_graph6_fig, f""" ###### Graphe complet""", f"""###### Kruskal""")

        # Etape 4
        st.write_stream(stream_data(f"""### Etape 4: Création de la matrice du graph de Kruskal""",0.02))
        st.caption("_**X2**_ est la matrice de distance pour le graphe minimal ( graph de Kruskal )")
        X2 = creerX2(st_graph6, saisieModifiee)
        st.write(X2)
        st.write_stream(stream_data(f"""Expliquer l'utilisation de -10000""",0.02))

        # Etape 5
        st.write_stream(stream_data(f"""### Etape 5: Création de la matrice X3""",0.02))
        st.latex(r"X3 = X1 \times X2")
        X3 = np.dot(X1, X2)
        st.write(X3)

        # Etape 6
        st.write_stream(stream_data(f"""### Etape 6: Création de la matrice aléatoire Pk""",0.02))
        st.caption("Pk est une matrice aléatoire unimodulaire dont la taille est égale à la longeur des noeuds du graphe minimal")
        st_taille_matrice = len(st_graph6.nodes())
        st_Pk = creerPk(st_taille_matrice)
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
        # Etape 3
        st.write_stream(stream_data(f"""### Etape 3: Création du nouveau graph en fonction de X2""",0.02))
        st_graph7, edges_with_weights = creerDecryptedGraph1(X2_from_keys)
        st_graph7_fig = afficherGraphe(st_graph7)
        st.pyplot(st_graph7_fig)
        st.write_stream(stream_data(f""" On peut remarquer que ce graph a la _**même forme que le graph de Kruskal**_. Toutefois, on peut remarquer que _**celui-ci n'a pas de lettre comme nom de sommet mais des numéros**_ qui _**représentent l'ordre des lettres**_""",0.02))
        # Etaoe 4
        st.write_stream(stream_data(f"""### Etape 4: Reconstruire le message""",0.02))
        st.write_stream(stream_data(f"""##### Initialisation du dictionnaire""",0.02))
        dico = create_dict_from_tuples(edges_with_weights, caractere_supplementaire)
        st.write(dico)

        st.write_stream(stream_data(f"""Commenter pourquoi A est la seule valeur du dico et pourquoi il y'a des null partout""",0.02))

        # Etape 5
        st.write_stream(stream_data(f"""### Etape 5: Reconstitution des valeurs manquantes du dictionnaire""",0.02)) 
        
        st.write_stream(stream_data(f"""Explication de l'algorithme""",0.02))
        # Etape 6
        st.write_stream(stream_data(f"""### Etape 6: Dictionnaire final et reconstitution du méssage""",0.02))
        st.toast('Méssage déchiffré avec succès', icon="🔓")
        # End

def display_matrix(matrix1, matrix2, matrix1_title, matrix2_title):
    col1, col2 = st.columns(2)

    with col1:
        st.write(matrix1_title, unsafe_allow_html=True)
        st.write(matrix1)
    with col2:
        st.write(matrix2_title, unsafe_allow_html=True)
        st.write(matrix2)
    


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

def streamlit_main():
    streamlit_process()

def get_graph_n(saisieUtilisateur, N):
    current_graph = 0 
    graph = None
    graph_fig = None
    while current_graph < N:
        if current_graph == 0:
            graph = creerGraphe1(saisieUtilisateur)
            graph_fig = afficherGraphe(graph)
        elif current_graph == 1:
            graph = creerGraphe2(graph)
            graph_fig = afficherGraphe(graph)
        elif current_graph == 2:
            graph = creerGraphe3(graph)
            graph_fig = afficherGraphe(graph)
        elif current_graph == 3:
            graph = creerGraphe4(graph)
            graph_fig = afficherGraphe(graph)
        elif current_graph == 4:
            graph = creerGraphe5(graph, "A", 0)
            graph_fig = afficherGraphe(graph)
        current_graph += 1
    return graph, graph_fig

def creerInversibleX1(position,saisieModifiee,graphe5):
    X1_temp = creerX(graphe5, saisieModifiee)
    while (position<len(saisieModifiee) and not estInversible(X1_temp)):
        position += 1
        saisieModifiee = saisieUtilisateur[:position] + caractere_supplementaire + saisieUtilisateur[position:]
        X1_temp = creerX(graphe5, saisieModifiee)
    return X1_temp

#  ------------------------------------------------------------ fin streamlit functions ------------------------------------------------------------ #

def main(position, caractere_supplementaire):
    ### Chiffrer ###

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
    # saisieModifiee = modifierComposants(saisieModifiee, caractere_supplementaire)

    # Créer la matrice de distance X1 pour graphe5
    #X1 = creerX1(graphe5, saisieModifiee, caractere_supplementaire)
    X1 = creerX(graphe5, saisieModifiee)

    print(estInversible(X1))
    print(X1)

    while (position<len(saisieModifiee) and not estInversible(X1)):
        position += 1
        graphe5 = creerGraphe5(graphe4, caractere_supplementaire, position)
        saisieModifiee = saisieUtilisateur[:position] + caractere_supplementaire + saisieUtilisateur[position:]
        X1 = creerX(graphe5, saisieModifiee)  
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
    dico = create_dict_from_tuples(edges_with_weights, caractere_supplementaire)
    print("Dico init:",dico)
    print("Chemins",edges_with_weights)
    
    while None in dico.values():
        dico, already_done = assign_values(dico, edges_with_weights, [])
        print("Dico:",dico)
        print("Already done:",already_done)


    #------------------

    plt.show()

if __name__ == "__main__":
    # caractere_supplementaire 
    caractere_supplementaire = 'A'
    #'►'
    position = 0
    # main(position, caractere_supplementaire)
    streamlit_main()