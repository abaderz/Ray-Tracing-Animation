# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import os
import imageio.v2 as imageio
from art import text2art

# Print ASCII art
print(text2art("RAY TRACING"))
print("[+] Bonjour ! Ce code générera des images et les sauvegardera dans un dossier appelé 'img' pour créer une animation.")
print("[+] Veuillez spécifier quelques variables pour ce code :")

# Nombre d'échantillons par pixel pour l'anti-aliasing.
echantillons = int(input("Nombre d'échantillons par pixel pour l'anti-aliasing (par défaut : 2): ") or 2)

# Nombres d'images que va générer le code.
frames = int(input("Nombre d'images que va générer le code pour l'animation (par défaut : 30): ") or 30)

# Nombres de spheres 
nb_spheres = int(input("Nombre de spheres que vous souhaitez créer (par défaut : 4): ") or 4)

def create_Ray(O, D):
    # créer un rayon 
    rayon = {
        'origine': np.array(O),
        'direction': np.array(D)
        }
    return rayon

def create_Sphere(P, r, amb, diff, spec, refl, m, i):
    # créer une spehere
    sphere = {
        "type": "sphere",
        "centre": np.array(P),
        "rayon": r,
        "ambiant":np.array(amb),
        "diffuse": np.array(diff),
        "specular": np.array(spec),
        "reflection": refl,
        "m": m,
        "index": i
    }
    return sphere
    
def create_Plane(P, n, amb, diff, spec, refl, m, i):
    # créer un plan 
    plan = {
        "type": "plan",
        "position": np.array(P),
        "normale": np.array(n),
        "ambiant": np.array(amb),
        "diffuse": np.array(diff),
        "specular": np.array(spec),
        "reflection": refl,
        "m": m,
        "index": int(i)
    }
    return plan


def normalize(x):
    # rend vecteur unitaire 
    norme = LA.norm(x)
    x = x/norme
    return x 

def rayAt(ray,t):
    # origine du rayon plus point de contact fois direction 
    D_rayon = ray['origine'] + t*ray["direction"]
    return D_rayon

def get_Normal(obj, M):
    if obj["type"] == "sphere":
        # La normale pour une sphère est la direction du point M par rapport au centre.
        return normalize(M - obj["centre"])
    elif obj["type"] == "plan":
        # La normale pour un plan est constante et égale à la normale du plan.
        return normalize(obj["normale"])

def intersect_Plane(ray, plane):

    Origine = ray['origine']
    Direction = ray['direction']
    Poisition = plane['position']
    Normale = plane['normale']
    
    # produit scalaire entre position et normale 
    A = np.dot((Poisition-Origine), Normale)

    # produit scalaire entre direction et normale 
    B = np.dot(Direction, Normale)

    # calcul de point d'intersection t
    t = A/B
    
    # cas ou il n'y as pas d'intersection
    if t> 0: 
        return t
    else : 
        return np.inf

def intersect_Sphere(ray, sphere):
    Origine = ray['origine']
    Direction = ray['direction']
    Centre = sphere['centre']
    Rayon = sphere['rayon']

    # Vecteur OC:
    oc = np.subtract(Origine, Centre)

    # Calcul du discriminant: 
    disc = (2 * np.dot(oc, Direction))**2 - 4 * (np.dot(Direction, Direction)) * (np.dot(oc, oc) -Rayon**2)

    # conditions limites:
    if disc <0:
        return np.inf
    
    # choix du point d'intersection le plus proche
    else : 
        t1 = (-(2 * np.dot(oc, Direction)) - np.sqrt(disc)) / (2 * (np.dot(Direction, Direction)))
        t2 = (-(2 * np.dot(oc, Direction)) + np.sqrt(disc)) / (2 * (np.dot(Direction, Direction)))
        if min(t1, t2) > 0:
            return min(t1, t2)
        else:
            return np.inf
        
def intersect_Scene(ray, obj):
    # Fonction qui calcule l'intersection entre un rayon et un objet de la scène.

    # Vérifie le type de l'objet
    if obj["type"] == "sphere":
        # Si l'objet est une sphère, utilise la fonction d'intersection pour les sphères
        t = intersect_Sphere(ray, obj)
    elif obj["type"] == "plan":
        # Si l'objet est un plan, utilise la fonction d'intersection pour les plans
        t = intersect_Plane(ray, obj)
    
    # Retourne le paramètre t d'intersection (t infini si pas d'intersection)
    return t

def Is_in_Shadow(obj_min, P, N):
    # Constante pour éviter l'acné (ombres des objets)
    acne_eps = 1e-4
    
    # Liste pour stocker les objets entre le point d'intersection et la source lumineuse
    L = []
    
    # Direction normalisée de P vers la source lumineuse
    lPL = normalize(Light['position'] - P)
    
    # Point décalé pour éviter l'acné
    PE = P + acne_eps * N
    
    # Créer le rayon pour tester les ombres
    rayTest = create_Ray(PE, lPL)
    
    # Parcourir tous les objets de la scène pour trouver les intersections
    for obj_test in scene:
        if obj_test["index"] != obj_min["index"]:
            # Calculer l'intersection entre rayTest et obj_test
            t_test = intersect_Scene(rayTest, obj_test)
            
            # Vérifier s'il y a une intersection entre le rayon et l'objet
            if t_test < np.inf:
                # Ajouter obj_test à la liste L
                L.append(obj_test)

    # Si la liste est vide, le point n'est pas dans l'ombre
    if not L:
        return False
    else:
        # Sinon, le point est dans l'ombre
        return True

    
# https://en.wikipedia.org/wiki/Specular_highlight  
def beckmann_distribution(theta_h, m):
    # Calcul de la distribution de Beckmann
    numerator = np.exp(-(np.tan(theta_h) / m)**2)
    denominator = np.pi * m**2 * np.cos(theta_h)**4
    return numerator / denominator


def eclairage(obj, light, P):
    # Vecteur normalisé de la lumière dirigé du point P vers la source lumineuse
    L = normalize(light['position'] - P)
    # Vecteur normalisé de la normale à la surface au point P
    N = get_Normal(obj, P)

    # Ka et la sont les couleurs ambiante de l'objet et de la lumière
    Ka = obj['ambiant']
    la = light['ambient']
    # Kd et ld sont les couleurs diffuses de l'objet et de la lumière
    Kd = obj['diffuse']
    ld = light['diffuse']
    # Ks et ls sont les couleurs spéculaires de l'objet et de la lumière
    Ks = obj['specular']
    ls = light['specular']

    # Direction de la lumière normalisée
    L = normalize(light['position'] - P)
    # Calcul du vecteur de réflexion R
    R = normalize(2 * np.dot(N, L) * N - L)
    # Vecteur normalisé de la vue dirigé du point P vers la caméra
    V = normalize(C - P)

    # Calcul du cosinus de l'angle entre la normale et la direction de la lumière
    cos_theta = np.dot(L, N) / (LA.norm(L) * LA.norm(N))
    # Calcul de l'angle entre R et V
    cos_alpha = np.dot(R, V) / (LA.norm(R) * LA.norm(V))

    # Assurer que le cosinus est positif (évite les problèmes avec les angles > 90 degrés)
    cos_theta = max(cos_theta, 0)
    # Assurer que le cosinus est positif
    cos_alpha = max(cos_alpha, 0)

    # Calcul de la couleur diffuse
    diffuse_color = Ka * la + Kd * ld * cos_theta

    # Ajouter la distribution de Beckmann au calcul de la couleur spéculaire
    beckmann = beckmann_distribution(np.arccos(np.dot(N, R)), materialShininess)

    # Calcul de la lumiere spéculaire
    specular_color = Ks * ls * cos_alpha**materialShininess * beckmann

    return diffuse_color + specular_color

def reflected_ray(dirRay, N):
    """
    Calculer la direction réfléchie d'un rayon incident par rapport à la normale à la surface.

    Args:
    - dirRay (numpy.array): La direction du rayon incident.
    - N (numpy.array): La normale à la surface au point d'intersection.

    Returns:
    - numpy.array: La direction du rayon réfléchi.
    """
    
    # Calculer le produit scalaire entre la direction du rayon incident et la normale
    dot_product = np.dot(dirRay, N)

    # Calculer la direction réfléchie en utilisant la formule : dirReflechi = dirRay - 2 * (dirRay . N) * N
    dirReflechi = normalize(dirRay - 2 * dot_product * N)

    # Retourner la direction du rayon réfléchi
    return dirReflechi


def compute_reflection(rayTest, depth_max, col):
    """
    Calculer la réflexion d'un rayon dans la scène en utilisant la technique du ray tracing.

    Args:
    - rayTest (dict): Le rayon à tracer.
    - depth_max (int): La profondeur maximale de la réflexion.
    - col (numpy.array): La couleur du pixel à la position actuelle.

    Returns:
    - numpy.array: La couleur réfléchie résultante.
    """
    
    # Initialiser le coefficient de réflexion à 1
    c = 1.0

    for _ in range(depth_max):

        # Récupérer obj, M, N, et col_ray en tracant le rayon dans la scène
        result_trace_ray = trace_ray(rayTest)

        if result_trace_ray is None:
            # Aucune intersection avec la scène, sortir de la boucle
            break

        obj, M, N, col_ray = result_trace_ray

        # Calculer la première intersection de rayTest avec la scène
        t = intersect_Scene(rayTest, obj)

        # Ajouter c * col_ray à col
        col += c * col_ray

        # Créer un point ME en décalant M de acne_eps dans la direction de N
        ME = M + acne_eps * N

        # Mettre à jour la direction du rayon réfléchi
        dirReflechi = reflected_ray(rayTest['direction'], N)

        # Mettre à jour rayTest avec le rayon réfléchi
        rayTest = create_Ray(ME, dirReflechi)

        # Mettre à jour le coefficient de réflexion
        c *= obj['reflection']

        # Mettre à jour le coefficient de rugosité
        m = obj.get('m', 0.2)

        # Calculer la distribution de Beckmann avec le nouvel angle
        beckmann = beckmann_distribution(np.arccos(np.dot(N, dirReflechi)), m)

        # Ajouter la distribution de Beckmann au calcul de la couleur réfléchie
        col += c * col_ray * beckmann

    return np.array(col)



def trace_ray(ray):
    """
    Suivre le rayon dans la scène et déterminer l'intersection la plus proche.

    Args:
    - ray (dict): Le rayon à suivre.

    Returns:
    - tuple: Un tuple contenant l'objet le plus proche, le point d'intersection, la normale à cet endroit,
             et la couleur du rayon à cet endroit.
    """
    # Initialiser les variables pour trouver l'objet le plus proche.
    tMin = np.inf
    objProche = None

    # Parcourir tous les objets de la scène pour trouver l'intersection la plus proche.
    for obj in scene:
        t = intersect_Scene(ray, obj)
        if t < tMin:
            tMin = t
            objProche = obj

    # Si le rayon n'intersecte aucun objet, retourner None.
    if objProche is None:
        return None

    # Calculer le point d'intersection et la normale à cet endroit.
    P = rayAt(ray, tMin)
    N = get_Normal(objProche, P)

    # Vérifier s'il y a de l'ombre
    shadow = Is_in_Shadow(objProche, P, N)

    if shadow:
        # Si le point est dans l'ombre, retourner None
        return None
    else:
        # Calculer la couleur diffuse
        diffuse_color = eclairage(objProche, Light, P)
        
        # Calculer la couleur de l'objet (la valeur de la clé ambiant pour l'instant).
        col_ray = objProche["ambiant"] * Light["ambient"] + diffuse_color

        return objProche, P, N, col_ray


# Taille de l'image
w = 800
h = 600
acne_eps = 1e-4
materialShininess = 50


img = np.zeros((h, w, 3), dtype=float)# image vide : que du noir
#Aspect ratio
r = float(w) / h
# coordonnées de l'écran : x0, y0, x1, y1.
S = (-1., -1. / r , 1., 1. / r )


# Position et couleur de la source lumineuse
Light = { 'position': np.array([5, 5, 0]),
          'ambient': np.array([0.05, 0.05, 0.05]),
          'diffuse': np.array([1, 1, 1]),
          'specular': np.array([1, 1, 1]) }

L = Light['position']
C = np.array([0., 0.1, 1.1])  # Coordonée du centre de la camera.
Q = np.array([0,0.3,0])  # Orientation de la caméra
img = np.zeros((h, w, 3)) # image vide : que du noir
materialShininess = 50
skyColor = np.array([0.321, 0.752, 0.850])
whiteColor = np.array([1,1,1])
depth_max = 10

# Création du dossier 'img' s'il n'existe pas déjà
os.makedirs('./img/', exist_ok=True)

# Trajet des boules selon l'axe x, de droite à gauche
trajet_x = list(np.linspace(-1., 2, 30))

# Trajet des boules selon l'axe y :
# - Partie 1 et 3 : quand ça monte
# - Partie 2 et 4 : quand ça descend
trajet_y = list(np.concatenate((np.linspace(0.6, -0.3, 7), np.linspace(-0.3, 0.6, 8), np.linspace(0.6, -0.3, 7), np.linspace(-0.3, 0.6, 8))))


for frame in range(frames):

    scene = [
        create_Sphere([trajet_x[frame]*(-1)**i, trajet_y[frame], -1-i*0.5], # Position
                    .6, # Rayon
                    np.array([i*0.2, i*0.2 * 0.6, i*0.2 * 0.]), # ambiant
                    np.array([1. , 0.6, 0. ]), # diffuse
                    np.array([1, 1, 1]), # specular
                    0.2, # reflection index
                    0.2, # coefficient de rugositéde beckman
                    1)
                    for i in range(0, nb_spheres)
    ] + [
        create_Plane([0., -.9, 0.], # Position
                    [0, 1, 0], # Normal
                    np.array([0.145, 0.584, 0.854]), # ambiant
                    np.array([0.145, 0.584, 0.854]), # diffuse
                    np.array([1, 1, 1]), # specular
                    0.7, # reflection index
                    0.2, # coefficient de rugositéde beckman
                    2) # index
    ]



    # Loop through all pixels.
    for i, x in enumerate(np.linspace(S[0], S[2], w)):
        if i % 10 == 0:
            print(i / float(w) * 100, "%")
        for j, y in enumerate(np.linspace(S[1], S[3], h)):

            # Créer le rayon correspondant à la vue de la caméra
            # C représente la position de la caméra
            # (x, y, 0) représente un point sur le plan de l'image en gros ca passe par chaque point un par un 
            # normalize(np.array([x, y, 0]) - C) vecteur normalisé de la direction du rayon moin l'origine
            ray = create_Ray(C, normalize(np.array([x, y, 0]) - C))
            
            # Gérer le cas où le rayon n'intersecte aucun objet (par exemple, définir une couleur de fond).
            col = np.array([0, 0, 0],dtype=float)  # Couleur de fond

            # Appeler la fonction compute_reflection pour calculer la couleur avec les réflexions
            col = compute_reflection(ray, depth_max, col)

            # boucle pour l'anti aliasing 
            for _ in range(echantillons): 
                # decalage du pixel principal 
                offset = np.random.uniform(-0.00125, 0.00125, 2) #uniforme pour que la moyenne des chiffre random soit 0 : (-0,00125 +0,00125)/2
                ray = create_Ray(C, normalize(np.array([x + offset[0], y + offset[1], 0]) - C)) # on calcul le rayon pour ses pixels 

                # Calculez la couleur pour chaque rayon et ajoutez-la à la couleur du pixel
                col = col + compute_reflection(ray, depth_max, np.zeros(3))
            
            # Moyennez les couleurs des échantillons pour obtenir la couleur finale du pixel
            col /= echantillons

            # La fonction clip permet de "forcer" les valeurs de col à rester dans la plage [0, 1].
            # on evite ca ValueError: Floating point image RGB values must be in the 0..1 range.
            img[h - j - 1, i] = np.clip(col, 0, 1) 
    

    plt.imsave(f'./img/photo{frame}.png', img)

    
# Spécifiez le chemin du dossier contenant vos images
dossier_images = './img/'

# Créez une liste pour stocker les noms de fichiers d'images
noms_images = [f'photo{i}.png' for i in range(30)]

# Créez une liste pour stocker les chemins complets des images
chemins_images = [os.path.join(dossier_images, nom) for nom in noms_images]

# Chargez les images dans une liste
images = [imageio.imread(chemin) for chemin in chemins_images]

# Spécifiez le chemin du fichier de sortie vidéo
chemin_video = 'animation.mp4'

# Utilisez imageio pour créer une vidéo à partir des images
with imageio.get_writer(chemin_video, fps=10) as writer:
    for chemin_image in chemins_images:
        img = imageio.imread(chemin_image)
        writer.append_data(img)

print(f'Vidéo créée avec succès : {chemin_video}')