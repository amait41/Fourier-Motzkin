# Exercice 1 : Fourier-Motzkin

On considère un problème de minimisation sous forme standard : <br>
<br>
$$
\left\{
    \begin{array}{ll}
        min f(x) = \sum\limits^{n}_{j=1} c_jx_j = c^Tx \\
        \sum\limits^{n}_{j=1} a_{i,j} x_j \leq b_i, \: i=1, \cdots, r \Leftrightarrow Ax\leq b \\
        x_j \geq 0, \: j=1, \cdots, n \Leftrightarrow x \geq 0 \\
    \end{array}
\right.
$$
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

# Exercice 2 : Problème de planification

Une entreprise, spécialisée dans la construction de cartes à puces, désire minimiser les coûts de
leurs productions. Lors de la production de ces cartes, l’entreprise a besoin de certains composants
électroniques, qu’elle produit elle-même. Une bonne planification de la production est facile à
réaliser puisque les composants sont produits en interne. Les trois composants dont la production
est à planifier sur quatre mois sont notés sous les références M1, M2, M3. Le coût de fabrication
de ces composants comporte les coûts de production proprements dits, les coûts de stockage, mais
aussi des coûts liés aux variations des niveaux en production. En effet, quand le niveau total (i.e.,
tout type de composants) de la production change, des réglages machines et des vérifications sont
à effectuer pour le mois en cours. Le coût associé est proportionnel à la quantité totale produite
en plus ou en moins par rapport au mois précédent. Le coût pour une augmentation est de 1 euros
par pièce en plus, alors que celui d’une diminution est seulement de 0.5 euros par pièce en moins.
Notons qu’un changement de niveau de production n’est autre que la différence entre la quantité
totale produite lors du mois en question et la quantité totale produite lors du mois précédent.
Les informations concernant, pour chaque composant, la demande (Dem) par période, les coûts
de production (en euros par unité) et de stockage (en euros par unité) ainsi que le stock initial et
le stock final désirés pour chacun des produits, sont données dans le tableau suivant :

|  |Dem. mois 1|Dem. mois 2|Dem. mois 3|Dem. mois 4|Coût prod.|Coût stock|Stock init|Stock final|
|--|-----------|-----------|-----------|-----------|----------|----------|----------|-----------|
|M1|1500       |3000       |2000       |4000       |20        |1         |10        |50         |
|M2|1300       |800        |800        |1000       |25        |2         |0         |10         |
|M1|2200       |1500       |2900       |1800       |10        |3         |50        |30         |

1. Ecrire le programme linéaire dont l’objectif est de fixer le plan de production qui minimise la somme des coûts de changement du niveau de production, des coûts de production et des coûts de stockage.
2. Ecrire le programme Python correspondant et calculer le résultat demandé.

## Modélisation

### Variables

Pour le mois $i$ et le composant $j$, on pose:
- $d_{i,j}$ la demande
- $x_{i,j}$ la production
- $s_{i,j}$ le stock.

De plus, nous aurons besoin de poser des contraintes sur le coût de manutention. On note alors $m_i$ le coût de manutantion pour les mois $i-1$ à $i$ tel que $i=2,3,4$ et $m_1=0$.

$x_{i,j}$, $s_{i,j}$ et $m_{i}$ seront les variables de notre système et $d_{i,j}$ des constantes.

### Fonction coût

La fonction est composée pour le mois $i$ du coût de production $C^{P}_{i}$, du coût de stockage $C^{S}_{i}$, et du coût de manutention $C^{M}_{i}$, tels que:

$$
\left\{
    \begin{array}{ll}
        C^P_i = 20x_{i,1} + 25x_{i,2} + 10x_{i,3} \\
        C^S_i = s_{i,1} + 2s_{i,2} + 3s_{i,3} \\
        C^M_i = m_i
    \end{array}
\right.
$$

Ainsi le coût total $C$ est définit comme suit:
$$
C = \sum\limits^4_{i=1} C^P_i + C^S_i + C^M_i
$$

### Contraintes

La production ajoutée aux stocks, doit être supérieure à la demande pour les mois 1 à 4 et pour les composants 1, 2 et 3 (1). Les stocks d'un mois pour un composant correspond exactement à la production du mois plus le stock  moins la demande du mois précédant (6). Les stocks initiaux (7) et finals (8) sont fixés. Les productions, les stocks et les coûts de maintenance sont positifs (resp. 4, 5, et 6). Et pour finir, on note pour les mois 2,3 et 4, $\Delta_i$ la variation de prodction du mois $i-1$ au mois $i$ tel que $\Delta_i = \sum\limits^3_{j=1} x_{i,j} - x_{i-1,j}, \: i=2,3,4$ (et $\Delta_1 = 0$). $\:$ Alors $m_i \geq max(-\frac{1}{2}\Delta_i, \Delta_i)$  (2 et 3).

Alors les contraintes sur les variables $x_{i,j}$, $s_{i,j}$ et $c_{i}$ se traduisent par le système suivant: <br>
<br>
$$
\left\{
    \begin{array}{ll}
        -x_{i,j} - s_{i,j} \leq -d_{i,j} \quad &\forall i = {1,\cdots,4}, \quad \forall j=1,\cdots,3 \quad &(1)\\
        - \frac{1}{2} \Delta_i - m_i \leq 0 \quad &\forall i = {1,\cdots,4} \quad &(2)\\
        \Delta_i - m_i \leq 0, \quad &\forall i = {1,\cdots,4} \quad &(3)\\
        -x_{i,j} \leq 0, \quad &\forall i = {1,\cdots,4}, \quad \forall j=1,\cdots,3 \quad &(4)\\
        -s_{i,j} \leq 0, \quad &\forall i = {1,\cdots,4}, \quad \forall j=1,\cdots,3 \quad &(5)\\
        -m_{i} \leq 0, \quad &\forall i = {1,\cdots,4} \quad &(6)\\
        x_{i-1,j} + s_{i-1,j} - d_{i-1,j} = s_{i,j} &\forall i = {1,\cdots,4} \quad &(7)\\
        s_{1,1} = 10, \: s_{1,2} = 0, \:s_{1,3} = 50 &\quad &(8)\\
        x_{4,1} + s_{4,1} - d_{4,1} = 50, \; x_{4,2} + s_{4,2} - d_{4,2} = 10, \: x_{4,3} + s_{4,3} - d_{4,3} = 30 &\quad &(9)\\
    \end{array}
\right.
$$

## Programmation

Nous allons à présent utiliser la fonction `linprog` du package `scipy.optimize` pour résoudre le problème.


```python
import numpy as np
import pandas as pd
from scipy.optimize import linprog
```

### Constantes


```python
# demande par mois par composant
d = np.array([[1500,1300,2200],
              [3000,800,1500],
              [2000,800,2900],
              [4000,1000,1800]])
# coût de production
cp = np.array([20,25,10])
# coût de stockage
cs = np.array([1,2,3])
# stock initial
stock_initial = np.array([10,0,50])
# stock final
stock_final = np.array([50,10,30])
```

### Contraintes d'inégalité


```python
# contrainte sur la demande
A_ub1 = -np.hstack([np.eye(12), np.eye(12), np.zeros((12,3))])
# contrainte sur les coûts de manutantion
A_ub2 = np.hstack([np.array([[1/2,1/2,1/2, -1/2,-1/2,-1/2, 0,0,0, 0,0,0],
                             [-1,-1,-1, 1,1,1, 0,0,0, 0,0,0],
                             [0,0,0, 1/2,1/2,1/2, -1/2,-1/2,-1/2, 0,0,0],
                             [0,0,0, -1,-1,-1, 1,1,1, 0,0,0],
                             [0,0,0, 0,0,0, 1/2,1/2,1/2, -1/2,-1/2,-1/2],
                             [0,0,0, 0,0,0, -1,-1,-1, 1,1,1]]),
                   np.zeros((6,12)),
                   -np.array([1,0,0, 1,0,0, 0,1,0, 0,1,0, 0,0,1, 0,0,1]).reshape(6,3)
                  ])
A_ub = np.vstack([A_ub1, A_ub2])

b_ub1 = -d.ravel()
b_ub2 = np.zeros(6)
b_ub = np.hstack([b_ub1, b_ub2])

print(A_ub.shape)
print(b_ub.shape)
```

    (18, 27)
    (18,)


### Contraintes d'égalité


```python
# contrainte sur le stock initial
A_eq1 = np.hstack([np.zeros((3,12)), np.eye(3), np.zeros((3,12))])
# contrainte sur les stocks intermédiaires
A_eq2 = np.hstack([np.eye(9), np.zeros((9,3)), np.eye(9,12) - np.eye(9,12,k=3), np.zeros((9,3))])
# contrainte sur le stock final
A_eq3 = np.hstack([np.zeros((3,9)), np.eye(3), np.zeros((3,9)), np.eye(3), np.zeros((3,3))])
A_eq = np.vstack([A_eq1, A_eq2, A_eq3])

b_eq1 = stock_initial
b_eq2 = d.ravel()[:9]
b_eq3 = stock_final + d.ravel()[9:]
b_eq = np.hstack([b_eq1, b_eq2, b_eq3])

print(A_eq.shape)
print(b_eq.shape)
```

    (15, 27)
    (15,)


### Fonction coût


```python
c_prod = np.hstack([cp, cp, cp, cp, np.zeros(15)])
c_stock = np.hstack([np.zeros(12), cs, cs, cs, cs, np.zeros(3)])
c_manu = np.hstack([np.zeros(24), np.ones(3)])
c = c_prod + c_stock + c_manu

print(c.shape)
```

    (27,)


### Résultats


```python
res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0,None), method="simplex")
print(res.message)
```

    Optimization terminated successfully.



```python
res.x = np.round(res.x,4)
prod = [[res.x[i%3+3*j] for i in range(3)] for j in range(4)]
stock = [[res.x[i%3+3*j+12] for i in range(3)] for j in range(4)]
```


```python
prod = pd.DataFrame({'Mois 1':prod[0], 'Mois 2':prod[1], 'Mois 3':prod[2],'Mois 4':prod[3]})
prod.index = ['Composant 1', 'Composant 2', 'Composant 3']
print('Production:')
display(prod)
```

    Production:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mois 1</th>
      <th>Mois 2</th>
      <th>Mois 3</th>
      <th>Mois 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Composant 1</th>
      <td>1670.0</td>
      <td>2820.0</td>
      <td>2595.0</td>
      <td>3455.0</td>
    </tr>
    <tr>
      <th>Composant 2</th>
      <td>1300.0</td>
      <td>800.0</td>
      <td>800.0</td>
      <td>1010.0</td>
    </tr>
    <tr>
      <th>Composant 3</th>
      <td>2150.0</td>
      <td>1500.0</td>
      <td>2900.0</td>
      <td>1830.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
stock_fin = [prod.iloc[0,:].sum() - d[:,0].sum() + 10,
             prod.iloc[1,:].sum() - d[:,1].sum() + 0,
             prod.iloc[2,:].sum() - d[:,2].sum() + 50,
            ]
stock = pd.DataFrame({'Mois 1':stock[0], 'Mois 2':stock[1], 'Mois 3':stock[2],'Mois 4':stock[3], 'Fin':stock_fin})
stock.index = ['Composant 1', 'Composant 2', 'Composant 3']
print('Stock:')
display(stock)
```

    Stock:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mois 1</th>
      <th>Mois 2</th>
      <th>Mois 3</th>
      <th>Mois 4</th>
      <th>Fin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Composant 1</th>
      <td>10.0</td>
      <td>180.0</td>
      <td>0.0</td>
      <td>595.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>Composant 2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Composant 3</th>
      <td>50.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
var = np.array([prod.iloc[:,0].sum(),
               prod.iloc[:,1].sum() - prod.iloc[:,0].sum(),
               prod.iloc[:,2].sum() - prod.iloc[:,1].sum(),
               prod.iloc[:,3].sum() - prod.iloc[:,2].sum()])
var = pd.DataFrame({'Variation production':var})
var.index = ['Mois 1', 'Mois 2', 'Mois 3', 'Mois 4']
display(var)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variation production</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mois 1</th>
      <td>5120.0</td>
    </tr>
    <tr>
      <th>Mois 2</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Mois 3</th>
      <td>1175.0</td>
    </tr>
    <tr>
      <th>Mois 4</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
print('Coût de total des 4 mois :',round(res.fun,2))
```

    Coût de total des 4 mois : 394460.0

