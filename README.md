# Fourier-Motzkin

<p>On considère un problème de minimisation sous forme standard :<br />
<br />
<span class="math display">\[\left\{
    \begin{array}{ll}
        min f(x) = \sum\limits^{n}_{j=1} c_jx_j = c^Tx \\
        \sum\limits^{n}_{j=1} a_{i,j} x_j \leq b_i, \: i=1, \cdots, r \Leftrightarrow Ax\leq b \\
        x_j \geq 0, \: j=1, \cdots, n \Leftrightarrow x \geq 0 \\
    \end{array}
\right.\]</span> &lt;br&gt; où <span class="math inline">\(x= (x_1, \cdots, x_n)^T \in \mathbb{R}^n, \: c=(c_1, \cdots, c_n)^T \in \mathbb{R}^n,\: b=(b_1, \cdots, b_r) \in \mathbb{R}^r, \: A \in \mathcal{M_{r,n}}(\mathbb{R})\)</span>.&lt;br&gt; La fonction ‘FourierMotzkin‘ qui retourne la liste <span class="math inline">\((min f, x_1, · · · , x_n)\)</span> par méthode de Fourier-Motzkin.</p>
