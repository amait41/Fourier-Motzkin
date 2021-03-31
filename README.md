# Fourier-Motzkin
On considère un problème de minimisation sous forme standard :
&lt;br&gt; &lt;br&gt;
$$\\left\\{
    \\begin{array}{ll}
        min f(x) = \\sum\\limits^{n}\_{j=1} c\_jx\_j = c^Tx \\\\
        \\sum\\limits^{n}\_{j=1} a\_{i,j} x\_j \\leq b\_i, \\: i=1, \\cdots, r \\Leftrightarrow Ax\\leq b \\\\
        x\_j \\geq 0, \\: j=1, \\cdots, n \\Leftrightarrow x \\geq 0 \\\\
    \\end{array}
\\right.$$
&lt;br&gt; où
*x* = (*x*<sub>1</sub>, ⋯, *x*<sub>*n*</sub>)<sup>*T*</sup> ∈ ℝ<sup>*n*</sup>, *c* = (*c*<sub>1</sub>, ⋯, *c*<sub>*n*</sub>)<sup>*T*</sup> ∈ ℝ<sup>*n*</sup>, *b* = (*b*<sub>1</sub>, ⋯, *b*<sub>*r*</sub>) ∈ ℝ<sup>*r*</sup>, *A* ∈ ℳ<sub>𝓇, 𝓃</sub>(ℝ).&lt;br&gt;
La fonction ‘FourierMotzkin‘ qui retourne la liste
(*m**i**n**f*, *x*<sub>1</sub>,  ·  · ·,*x*<sub>*n*</sub>) par méthode
de Fourier-Motzkin.
