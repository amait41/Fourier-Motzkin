# Fourier-Motzkin

On considère un problème de minimisation sous forme standard : <br>
<br>
\\[
\left\{
    \begin{array}{ll}
        min f(x) = \sum\limits^{n}_{j=1} c_jx_j = c^Tx \\
        \sum\limits^{n}_{j=1} a_{i,j} x_j \leq b_i, \: i=1, \cdots, r \Leftrightarrow Ax\leq b \\
        x_j \geq 0, \: j=1, \cdots, n \Leftrightarrow x \geq 0 \\
    \end{array}
\right.
\\]
<br>
où $x= (x_1, \cdots, x_n)^T \in \mathbb{R}^n, \: c=(c_1, \cdots, c_n)^T \in \mathbb{R}^n,\: b=(b_1, \cdots, b_r) \in \mathbb{R}^r, \: A \in \mathcal{M_{r,n}}(\mathbb{R})$.<br>
Ecrivons une fonction Python `FourierMotzkin(A,b,c)` qui retourne la liste $(min f, x_1, · · · , x_n)$
par méthode de Fourier-Motzkin.

Dans un premier temps on définit le premier coefficient non-nul de notre vecteur $c=(c_1, \cdots, c_n)^T$ et on note son indice $k$. Si $c$ n'est pas nul alors on effectue le changement de variable (2) suivant: <br>
<br>
$$
\left\{
    \begin{array}{ll}
        x_1 &= x_1 \\
            &\vdots\\
        x_{k-1} &= x_{k-1} \\
        u &= c_kx_k + \cdots + c_nx_n \\
        x_{k+1} &= x_{k+1} \\
            &\vdots \\
        x_n &= x_n \\
    \end{array}
\right.
$$
<br>
puis on échange la colonne sur laquelle nous avons fait le changement de variable avec la dernière. En suite, on concatène sur l'axe vertical $-I_n$ à $A$ (3) et $\underbrace{(0, \cdots, 0)}_{n fois}$ à $b$ (3 bis) pour établir les contraintes de positivité.

Puis viens la phase de descente (4) suivit de la phase de rémontée (5) toutes deux expliquées ci-dessous.


```python
import numpy as np
```


```python
def FourierMotzkin(A, b, c):
    
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    n = len(c) # nombre de variables
    
    k = premier_coef_non_nul(c)   #(1)
    if k == None:
        raise ValueError('Le vecteur c est nul.')

    B = changement_variable(A, c, k)       # (2)
    # ajout des constantes dans le système
    b = np.hstack((b, np.zeros(A.shape[1])))   # (3 bis)
    
    stock_fa = descente(B, b, n)  # (4)
    
    return remontee(stock_fa, c, n, k) # (5)
```


```python
def premier_coef_non_nul(vect):
    '''
    Retourne l'indice du premier coefficient non-null d'un vecteur.
    '''
    for i in range(len(vect)):
        if vect[i] != 0:
            return i
```


```python
def changement_variable(A, c, k):
    '''
    Args : matrice A de taille n*r, vecteur c de taile 1*r, 
    k indice du premier coef non nul de c
    
    Return : La matrice de taille (r+n)*r correspond au
    changement de variable sur x_k par u = ck*xk + ... + cn*xn 
    (où k est le premier coefficient non-null de c) sur A, 
    concaténée avec -I_n.
    '''
    # ajout de la contrainte de positivité sur les variables
    (r,n) = A.shape
    A = np.vstack((A, -np.eye(n))) # (3)
    
    # changement de variable u
    for i in range(r+n):
        for j in range(n):
            if j != k:
                A[i,j] -= A[i,k]*c[j] / c[k]
            else:
                A[i,j] /= c[k]
    
    # on échange la colonne k avec la dernière 
    tmp = np.copy(A)
    A[:,k], A[:,-1] = A[:,-1], tmp[:,k]
    
    return A
```

La phase de descente consiste à projeter l'espace définit par $Ax \leq b$ sur les variables $x_2, \cdots, x_n, u$ puis sur les variables $x_3, \cdots, x_n, u$ et ainsi de suite jusqu'à obtenir un encadrement de $u$. La fonction *bornes* nous permet de stocker à chaque projection les formes affines qui encadrent la variable à éliminer.


```python
def descente(A, b, n):
    """
    Args :
    A matrice définissant un ensemble de points dans R^n et b un 
    vecteur correspondant aux  bornes supérieures des contraintes.
    
    Return :
    Liste de listes contenant les intervalles 
    [max(formes affines), min(formes affines)] bornant les 
    variables éliminées pour les projections de l'espace donnée.
    """
    # on concatène A et -b.reshape((r,1))
    A = np.c_[A,-b]
    A = norm(A)
    stock_fa = []
    for i in range(n):
        stock_fa.append(bornes(A, n))
        A = norm(proj(A))
    return stock_fa
```


```python
def norm(B):
    """
    Retourne la matrice normalisé telle qu'il est uniquement 
    des -1, 0, ou 1 sur la première colonne.
    """
    (r,n) = B.shape
    for i in range(r):
        if B[i,0] != 0 and abs(B[i,0]) != 1:
            B[i,:] /= abs(B[i,0])
    return B
```


```python
def proj(C):
    """
    Args :
    Matrice normalisée (cf norm).
    
    Return :
    Système définissant l'espace projeté sur les variables 
    x_i, ..., x_n où i correspond à la deuxième colonne
    de la matrice donnée.
    """
    (p,q) = np.shape(C)
    E = np.array([]).reshape(0,q-1)
    G = np.array([]).reshape(0,q-1)
    D = np.array([]).reshape(0,q-1)
    
    for i in range(p):
        line = np.copy(C[i][1:]).reshape(1,q-1)
        if C[i][0] > 0:
            D = np.vstack((D,-line))
        elif C[i][0] < 0:
            G = np.vstack((G,line))
        else:
            E = np.vstack((E,line)) 
    
    for g in G:
        for d in D:
            E = np.vstack((E,(g-d).reshape(1,q-1)))
    
    return E
```


```python
def bornes(A, n):
    """
    Entrée : La matrice A normée définissant un polyèdre et 
    le nombre de variables du système complet (ie non projeté).

    Sortie : L'intervalle [max(formes affines), min(formes affines)] 
    bornant la première variable du système de contraintes Ax - b <= 0
    sous la forme d'une liste de listes de coefficients des formes affines.
    """
    stock_fa = [[],[]]
    nb_lignes, q = np.shape(A)[0],np.shape(A)[1]-1
    # q = nombre de variables du système courant
    # n = nombre de variables du système complet

    for i in range(nb_lignes):
        if A[i,0] > 0: # x1 + forme affine < 0
            forme_affine = [0 for k in range(n-q+1)] + list(-A[i,1:])
            stock_fa[1].append(forme_affine)

        elif A[i,0] < 0: # -x1 + forme affine < 0
            forme_affine = [0 for k in range(n-q+1)] + list(A[i,1:])
            stock_fa[0].append(forme_affine)

    #On regarde si la variable est non bornée
    if stock_fa[0]==[]:
        stock_fa[0]=[[0 for i in range(n)] + [-1e16]]
    if stock_fa[1]==[]:
        stock_fa[1]=[[0 for i in range(n)] + [1e16]]

    return stock_fa
```

Pour finir, on utilise la fonction *remontee* qui va nous permettre de connaitre à la fois le minimum $f(x)$ tq $Ax \leq b$ et aussi un point $x^*=(x_1^*, \cdots, x_n^*)$ pour lequel le minimum est atteint.

Pour ce faire, on détermine le minimum de $u$ en prennant le max des constantes dans la première liste.

Pour calculer la valeur des autres variables, on regarde le signe devant la variable en question dans le vecteur $c$. Si le coefficient est positif alors on prend le minimum des formes affines dans lesquelles on aura remplacé les variables pour les minimums précédement trouvés. Si le coefficient est négatif alors on fait de même mais en prennant le maximum.

Enfin, on calcul la valeur de la variable $x_k$ que nous avions remplacée puis on remet les minimums dans le bon ordre. On retourne le résultat sous la forme $(\underset{Ax \leq b}{\min} f(x), x_1^*, \cdots, x_n^*)$.


```python
def remontee(stock_fa, c, n, k):
    """
    Args:
    Stock des matrices représentant les contraintes,
    projection après projection.
    
    Return :
    Liste de la forme [min c.dot(x), x_1, ..., x_n]
    """
    bornes_inf = [0 for i in range(n)]

    # on détermine le minimum de u
    liste_inf = stock_fa[-1][0]
    inf = max([liste_inf[i][n] for i in range(len(liste_inf))] )
    min_f = inf
    bornes_inf[-1]=inf
    
    # on dertermine le minimum des variables différantes de x_k
    for g in range(n-2,-1,-1):
        # Si le coefficient devant x_k est positif, 
        # on prend le minimum des formes affines qui majorent,
        if c[g]>0:
            list_inf = stock_fa[g][0]
            inf = max([sum([bornes_inf[j] * list_inf[i][j] for j in range(n)]) + list_inf[i][n] for i in range(len(list_inf))])
        # sinon on prend le maximum le formes affines qui minorent.
        else :
            list_inf = stock_fa[g][1]
            inf = min([sum([bornes_inf[j] * list_inf[i][j] for j in range(n)]) + list_inf[i][n] for i in range(len(list_inf))])
        bornes_inf[g] = inf
    
    # on détermine min(x_k)
    c_aux = list(np.copy(c))
    c_aux[k] = c[-1]
    c_aux[-1] = c[k]
    min_k = (bornes_inf[-1] - sum([c_aux[i]*bornes_inf[i] for i in range(n-1)])) / c[k] 
    
    # on rétablie dans le bon ordre les minimums 
    bornes_inf[n-1] = bornes_inf[k]
    bornes_inf[k] = min_k
    return [min_f] + bornes_inf
```
