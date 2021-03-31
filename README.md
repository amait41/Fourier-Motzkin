# Fourier-Motzkin

<p>On considère un problème de minimisation sous forme standard : <br>
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
Ecrivons une fonction Python <code>FourierMotzkin(A,b,c)</code> qui retourne la liste $(min f, x_1, · · · , x_n)$
par méthode de Fourier-Motzkin.</p>
