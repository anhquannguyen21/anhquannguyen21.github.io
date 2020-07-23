---
layout: post
title: Các phương pháp Matrix Decomposition (Phần 3)
tags: [Linear Algebra]
---

Ở bài trước mình đã nói về Cholesky Decomposition, bài này mình sẽ trình bày về QR Decomposition.
# I. Các khái niệm cơ bản
## 1. Tích vô hướng và không gian Euclid 
- Cho <img src="https://i.upmath.me/svg/V" alt="V" /> là không gian vector. Ánh xạ  
<img src="https://i.upmath.me/svg/%5Clangle%2C%20%5Crangle%3A%20V%20%5Ctimes%20V%20%5Crightarrow%20%5Cmathbb%7BR%7D%20%20" alt="\langle, \rangle: V \times V \rightarrow \mathbb{R}  " />  
 <img src="https://i.upmath.me/svg/%5Chspace%7B3mm%7D%20%20" alt="\hspace{3mm}  " /> <img src="https://i.upmath.me/svg/(u%2C%20v)%20%5Crightarrow%20%5Clangle%20u%2C%20v%5Crangle%20" alt="(u, v) \rightarrow \langle u, v\rangle " />  
được gọi là tích vô hướng trong <img src="https://i.upmath.me/svg/V" alt="V" /> nếu với mọi <img src="https://i.upmath.me/svg/u%2C%20v%2C%20w%20%5Cin%20V" alt="u, v, w \in V" /> và <img src="https://i.upmath.me/svg/%5Calpha%2C%20%5Cbeta%20%5Cin%20%5Cmathbb%7BR%7D" alt="\alpha, \beta \in \mathbb{R}" />, thỏa các tính chất sau:  
<img src="https://i.upmath.me/svg/%5Ctextbf%7B(i)%7D%20%5Clangle%20%5Calpha%20u%20%2B%5Cbeta%20v%2C%20w%20%5Crangle%20%3D%20%5Calpha%5Clangle%20u%2C%20w%20%5Crangle%20%2B%20%5Cbeta%5Clangle%20v%2C%20w%20%5Crangle%20%20" alt="\textbf{(i)} \langle \alpha u +\beta v, w \rangle = \alpha\langle u, w \rangle + \beta\langle v, w \rangle  " />;  
<img src="https://i.upmath.me/svg/%5Ctextbf%7B(ii)%7D%20%5Clangle%20w%2C%20%5Calpha%20u%20%2B%5Cbeta%20v%5Crangle%20%3D%20%5Calpha%5Clangle%20u%2C%20w%20%5Crangle%20%2B%20%5Cbeta%5Clangle%20v%2C%20w%20%5Crangle%20%20" alt="\textbf{(ii)} \langle w, \alpha u +\beta v\rangle = \alpha\langle u, w \rangle + \beta\langle v, w \rangle  " />;   
<img src="https://i.upmath.me/svg/%20%5Ctextbf%7B(iii)%7D%20%5Clangle%20u%2C%20v%5Crangle%3D%20%5Clangle%20v%2C%20u%5Crangle" alt=" \textbf{(iii)} \langle u, v\rangle= \langle v, u\rangle" />;  
<img src="https://i.upmath.me/svg/%5Ctextbf%7B(iv)%7D%20%5Clangle%20u%2C%20u%5Crangle%5Cge%200" alt="\textbf{(iv)} \langle u, u\rangle\ge 0" /> trong đó <img src="https://i.upmath.me/svg/%5Clangle%20u%2C%20u%5Crangle%20%3D%200%20%5Ciff%20u%3D0" alt="\langle u, u\rangle = 0 \iff u=0" />.  
Định nghĩa. Ta gọi một không gian vector hữu hạn chiều với tích vô hướng là một không gian Euclid.  
Cho không gian vector <img src="https://i.upmath.me/svg/V%20%3D%20%5Cmathbb%7BR%7D%5En" alt="V = \mathbb{R}^n" /> với <img src="https://i.upmath.me/svg/u%3D(x_1%2C%20x_2%2C...%2Cx_n)%2C%20v%3D(y_1%2C%20y_2%2C..%2C%20y_n)" alt="u=(x_1, x_2,...,x_n), v=(y_1, y_2,.., y_n)" />, ta định nghĩa <img src="https://i.upmath.me/svg/%5Clangle%20u%2C%20v%5Crangle%3A%3D%20x_1y_1%2Bx_2y_2%2B...%2Bx_ny_n" alt="\langle u, v\rangle:= x_1y_1+x_2y_2+...+x_ny_n" />.  
Khi đó <img src="https://i.upmath.me/svg/V" alt="V" /> là không gian Euclid. Tích vô hướng này được gọi là tích vô hướng chính tắc trong <img src="https://i.upmath.me/svg/%5Cmathbb%7BR%7D%5En" alt="\mathbb{R}^n" />.  
- [Bất đẳng thức Cauchy - Schwarz]. Với mọi <img src="https://i.upmath.me/svg/u%2C%20v%20%5Cin%20V" alt="u, v \in V" /> ta có  
<img src="https://i.upmath.me/svg/%20%5Clangle%20u%2C%20v%5Crangle%20%5Cle%20%5C%7Cu%20%5C%7C%5E2%20%5C%7Cv%5C%7C%5E2%20" alt=" \langle u, v\rangle \le \|u \|^2 \|v\|^2 " />.  
Dấu <img src="https://i.upmath.me/svg/%3D" alt="=" /> xảy ra khi và chỉ khi <img src="https://i.upmath.me/svg/u%2C%20v" alt="u, v" /> phụ thuộc tuyến tính.  
- [Bất đẳng thức tam giác]. Với mọi <img src="https://i.upmath.me/svg/u%2C%20v%20%5Cin%20V" alt="u, v \in V" /> ta có   
<img src="https://i.upmath.me/svg/%5C%7Cu%2Bv%20%5C%7C%20%5Cle%20%5C%7Cu%5C%7C%20%2B%20%5C%7Cv%20%5C%7C" alt="\|u+v \| \le \|u\| + \|v \|" />.  
Dấu <img src="https://i.upmath.me/svg/%3D" alt="=" /> xẩy ra khi và chỉ khi tồn tại <img src="https://i.upmath.me/svg/%5Clambda%20%5Cge%200" alt="\lambda \ge 0" /> sao cho <img src="https://i.upmath.me/svg/v%3D%20%5Clambda%20u" alt="v= \lambda u" />.  
- Định nghĩa. Cho <img src="https://i.upmath.me/svg/V" alt="V" /> là không gian Euclid và <img src="https://i.upmath.me/svg/u%2C%20v%20%5Cin%20V" alt="u, v \in V" />. Góc giữa hai vector <img src="https://i.upmath.me/svg/u%2C%20v" alt="u, v" /> là <img src="https://i.upmath.me/svg/%5Ctheta%20%20%5Cin%5B0%2C%20%5Cpi%5D" alt="\theta  \in[0, \pi]" /> thỏa <img src="https://i.upmath.me/svg/cos%5Ctheta%3D%20%5Cdfrac%7B%5Clangle%20u%2C%20v%5Crangle%7D%7B%5C%7Cu%5C%7C%5C%7Cv%5C%7C%7D" alt="cos\theta= \dfrac{\langle u, v\rangle}{\|u\|\|v\|}" />
## 2. Cơ sở trực giao và cơ sở trực chuẩn, quá trình trực chuẩn hóa Gram-Schmidt  
Giả sử trên <img src="https://i.upmath.me/svg/V" alt="V" /> trang bị một tích vô hướng. Hai vector được gọi là trực giao, ký hiệu <img src="https://i.upmath.me/svg/x%20%5Cperp%20y" alt="x \perp y" /> nếu <img src="https://i.upmath.me/svg/%5Clangle%20x%2C%20y%5Crangle%3D0" alt="\langle x, y\rangle=0" />.  
- Hệ vector <img src="https://i.upmath.me/svg/%5C%7Be_1%2C%20e_2%2C...%2Ce_m%5C%7D" alt="\{e_1, e_2,...,e_m\}" /> trong <img src="https://i.upmath.me/svg/V" alt="V" /> gọi là hệ trực giao nếu <img src="https://i.upmath.me/svg/e_i%20%5Cne%200" alt="e_i \ne 0" /> và <img src="https://i.upmath.me/svg/e_i%20%5Cperp%20e_j" alt="e_i \perp e_j" /> với mọi <img src="https://i.upmath.me/svg/i%20%5Cne%20j%2C%20i%2C%20j%3D%20%5Coverline%7B1%2Cm%7D" alt="i \ne j, i, j= \overline{1,m}" />, tức là  
<img src="https://i.upmath.me/svg/%20%5Clangle%20e_i%2C%20e_j%20%5Crangle%3D%200" alt=" \langle e_i, e_j \rangle= 0" /> nếu <img src="https://i.upmath.me/svg/i%20%5Cne%20j" alt="i \ne j" /> và <img src="https://i.upmath.me/svg/%20%5Clangle%20e_i%2C%20e_j%20%5Crangle%20%3D%20%5C%7Ce_i%5C%7C%5E2" alt=" \langle e_i, e_j \rangle = \|e_i\|^2" /> nếu <img src="https://i.upmath.me/svg/i%3Dj" alt="i=j" />.  
- Hệ trực giao mà <img src="https://i.upmath.me/svg/%5C%7Ce_i%5C%7C%3D1" alt="\|e_i\|=1" /> gọi là hệ trực chuẩn.  
- Hệ trực giao là hệ độc lập tuyến tính.  
Cơ sở <img src="https://i.upmath.me/svg/e%3D(e_1%2C%20e_2%2C...%2Ce_n)" alt="e=(e_1, e_2,...,e_n)" /> gọi là một cơ sở trực giao (trực chuẩn) nếu nó là một hệ trực giao (trực chuẩn).  
- Định lý [Gram - Schmidt]. Nếu hệ các vector <img src="https://i.upmath.me/svg/%5C%7Bv_1%2Cv_2%2C...%2Cv_m%5C%7D" alt="\{v_1,v_2,...,v_m\}" /> là độc lập tuyến tính bất kỳ trong <img src="https://i.upmath.me/svg/V" alt="V" /> thì tồn tại hệ trực chuẩn <img src="https://i.upmath.me/svg/%5C%7Be_1%2Ce_2%2C...%2Ce_m%20%5C%7D" alt="\{e_1,e_2,...,e_m \}" /> sao cho <img src="https://i.upmath.me/svg/e_i%20%5Cin%20span%5C%7Bv_1%2C...%2Cv_i%5C%7D%2C%20i%3D%20%5Coverline%7B1%2Cm%7D" alt="e_i \in span\{v_1,...,v_i\}, i= \overline{1,m}" />.  
Ta chứng minh bằng quy nạp.  
Với <img src="https://i.upmath.me/svg/j%3D1" alt="j=1" /> đặt <img src="https://i.upmath.me/svg/e_1%3D%5Cdfrac%7Bv_1%7D%7B%5C%7Cv_1%5C%7C%7D" alt="e_1=\dfrac{v_1}{\|v_1\|}" />. Giả sử xây dựng được hệ <img src="https://i.upmath.me/svg/%5C%7Be_1%2C...%2Ce_j%5C%7D" alt="\{e_1,...,e_j\}" /> trực chuẩn và <img src="https://i.upmath.me/svg/e_k%20%5Cin%20span%5C%7Bv_1%2C...%2Cv_k%5C%7D%2C%20%5Cforakk%20k%20%5Cin%20%5Coverline%7B1%2Cj%7D" alt="e_k \in span\{v_1,...,v_k\}, \forakk k \in \overline{1,j}" />. Ta chỉ ra cách cây dựng <img src="https://i.upmath.me/svg/e_%7Bj%2B1%7D" alt="e_{j+1}" />.  
Đặt <img src="https://i.upmath.me/svg/e'_%7Bj%2B1%7D%3Dv_%7Bj%2B1%7D%2B%5Calpha_1e_1%2B...%2B%5Calpha_je_j" alt="e'_{j+1}=v_{j+1}+\alpha_1e_1+...+\alpha_je_j" />, các <img src="https://i.upmath.me/svg/%5Calpha_i" alt="\alpha_i" /> xác định sau và <img src="https://i.upmath.me/svg/e'_%7Bj%2B1%7D%20%5Cperp%20e_k%2C%20%5Cforakk%20k%20%5Cin%20%5Coverline%7B1%2Cj%7D%20" alt="e'_{j+1} \perp e_k, \forakk k \in \overline{1,j} " />.  
Ta có <img src="https://i.upmath.me/svg/%5Clangle%20e'_%7Bj%2B1%7D%2C%20e_k%20%5Crangle%3D0%20%5Ciff%20%5Clangle%20v_%7Bj%2B1%7D%2C%20e_k%20%5Crangle%20%2B%5Calpha_k%20%5C%7Ce_k%5C%7C%5E2%3D0%20" alt="\langle e'_{j+1}, e_k \rangle=0 \iff \langle v_{j+1}, e_k \rangle +\alpha_k \|e_k\|^2=0 " />.  
Mà <img src="https://i.upmath.me/svg/%5C%7Ce_k%5C%7C%3D1" alt="\|e_k\|=1" /> nên <img src="https://i.upmath.me/svg/%5Calpha_k%3D-%5Clangle%20v_%7Bj%2B1%7D%2C%20e_k%20%5Crangle%20%20" alt="\alpha_k=-\langle v_{j+1}, e_k \rangle  " /> hay <img src="https://i.upmath.me/svg/%5Cdisplaystyle%20e'_%7Bj%2B1%7D%3Dv_%7Bj%2B1%7D%20-%5Csum_%7Bk%3D1%7D%5Ej%20%5Clangle%20v_%7Bj%2B1%7D%2C%20e_k%20%5Crangle%20e_k%20" alt="\displaystyle e'_{j+1}=v_{j+1} -\sum_{k=1}^j \langle v_{j+1}, e_k \rangle e_k " />  
Vì <img src="https://i.upmath.me/svg/%5C%7Bv_1%2C...%2C%20v_j%2C%20v_%7Bj%2B1%7D%5C%7D" alt="\{v_1,..., v_j, v_{j+1}\}" /> độc lập tuyến tính nên <img src="https://i.upmath.me/svg/v_%7Bj%2B1%7D" alt="v_{j+1}" /> không biểu diễn tuyến tính qua <img src="https://i.upmath.me/svg/%5C%7Be_1%2C...%2Ce_j%5C%7D" alt="\{e_1,...,e_j\}" />, vì <img src="https://i.upmath.me/svg/span%5C%7Be_1%2C...%2Ce_j%5C%7D%3D%20span%5C%7Bv_1%2C..%2C%20v_j%5C%7D" alt="span\{e_1,...,e_j\}= span\{v_1,.., v_j\}" />, do đó <img src="https://i.upmath.me/svg/e_%7Bj%2B1%7D%5Cne%200" alt="e_{j+1}\ne 0" />.  
Đặt <img src="https://i.upmath.me/svg/e_%7Bj%2B1%7D%3D%5Cdfrac%7Be'_%7Bj%2B1%7D%7D%7B%5C%7Ce'_%7Bj%2B1%7D%5C%7C%7D" alt="e_{j+1}=\dfrac{e'_{j+1}}{\|e'_{j+1}\|}" />. Ta được hệ trực chuẩn thỏa mãn định lý.  
- Hệ quả: Nếu <img src="https://i.upmath.me/svg/v%3D(v_1%2C...%2C%20v_n)" alt="v=(v_1,..., v_n)" /> là một cơ sở bất kỳ của <img src="https://i.upmath.me/svg/V" alt="V" /> thì tồn tại cơ sở trực chuẩn <img src="https://i.upmath.me/svg/e%3D(e_1%2C..%2Ce_n)" alt="e=(e_1,..,e_n)" /> sao cho <img src="https://i.upmath.me/svg/e_j%20%5Cin%20span%5C%7Bv_1%2C..%2Cv_j%5C%7D" alt="e_j \in span\{v_1,..,v_j\}" />.  
- Định lý: Nếu <img src="https://i.upmath.me/svg/e%3D(e_1%2C...%2Ce_n)" alt="e=(e_1,...,e_n)" /> là một cơ sở trực chuẩn của <img src="https://i.upmath.me/svg/V" alt="V" /> thì với mọi <img src="https://i.upmath.me/svg/x%20%5Cin%20V" alt="x \in V" /> ta có <img src="https://i.upmath.me/svg/%5Cdisplaystyle%20x%3D%5Csum_%7Bi%3D1%7D%5En%5Clangle%20x%2C%20e_i%20%5Crangle%20e_i" alt="\displaystyle x=\sum_{i=1}^n\langle x, e_i \rangle e_i" />.  
Giả sử <img src="https://i.upmath.me/svg/x%3Dx_1e_1%2Bx_2e_2%2B...%2Bx_ne_n" alt="x=x_1e_1+x_2e_2+...+x_ne_n" />.  
<img src="https://i.upmath.me/svg/%5Cimplies%20%5Clangle%20x%2C%20e_i%20%5Crangle%3Dx_1%5Clangle%20e_1%2C%20e_i%20%5Crangle%2B%20x_2%5Clangle%20e_2%2C%20e_i%20%5Crangle%2B...%2B%20x_n%5Clangle%20e_n%2C%20e_i%20%5Crangle" alt="\implies \langle x, e_i \rangle=x_1\langle e_1, e_i \rangle+ x_2\langle e_2, e_i \rangle+...+ x_n\langle e_n, e_i \rangle" />.  
Vì <img src="https://i.upmath.me/svg/e" alt="e" /> là cơ sở trực chuẩn nên <img src="https://i.upmath.me/svg/%5Clangle%20u_i%2C%20u_j%20%5Crangle%3D0" alt="\langle u_i, u_j \rangle=0" /> nếu <img src="https://i.upmath.me/svg/i%5Cne%20j" alt="i\ne j" /> và <img src="https://i.upmath.me/svg/%5Clangle%20u_i%2C%20u_j%20%5Crangle%3D1" alt="\langle u_i, u_j \rangle=1" /> nếu <img src="https://i.upmath.me/svg/i%20%3D%20j" alt="i = j" />.  
Suy ra <img src="https://i.upmath.me/svg/%5Clangle%20u_i%2C%20u_j%20%5Crangle%3Dx_i" alt="\langle u_i, u_j \rangle=x_i" />.  
Ví dụ: Trực chuẩn hóa Gram - Schmidt hệ vector  
<img src="https://i.upmath.me/svg/v_1%3D(1%2C1%2C1)%2C%20v_2%3D(0%2C1%2C1)%2C%20v_3%3D(0%2C0%2C1)" alt="v_1=(1,1,1), v_2=(0,1,1), v_3=(0,0,1)" />
[![qrdecomposition.png](https://i.postimg.cc/J7ZCy3Mz/qrdecomposition.png)](https://postimg.cc/MX6FNQnC)

# II. Phân tích QR
## 1. Phát biểu 
- Giả sử rằng <img src="https://i.upmath.me/svg/A%20%5Cin%20M_%7Bn%20%5Ctimes%20m%7D(%5Cmathbb%7BR%7D)" alt="A \in M_{n \times m}(\mathbb{R})" /> là ma trận gồm <img src="https://i.upmath.me/svg/m" alt="m" /> cột độc lập tuyến tính, <img src="https://i.upmath.me/svg/A" alt="A" /> viết dưới dạng các vector cột là <img src="https://i.upmath.me/svg/A%3D(v_1%2C%20v_2%2C...%2Cv_m)" alt="A=(v_1, v_2,...,v_m)" />. Trực chuẩn hóa các vector <img src="https://i.upmath.me/svg/v_1%2C%20v_2%2C...%2C%20v_m" alt="v_1, v_2,..., v_m" /> ta được các vector <img src="https://i.upmath.me/svg/e_1%2C%20e_2%2C...%2C%20e_m" alt="e_1, e_2,..., e_m" />. Mặc khác từ định lý nêu ở trên ta có  
<img src="https://i.upmath.me/svg/%5Cdisplaystyle%20v_k%3D%5Csum_%7Bi%3D1%7D%5Ek%20%5Clangle%20v_k%2C%20e_i%20%5Crangle%20e_i" alt="\displaystyle v_k=\sum_{i=1}^k \langle v_k, e_i \rangle e_i" />  
Vì vậy ta có   
<img src="https://i.upmath.me/svg/A%3D(v_1%2C%20v_2%2C...%2C%20v_m)%3D(e_1%2C%20e_2%2C...%2C%20e_m)%5Cleft(%20%5Cbegin%7Bmatrix%7D%20%0A%5Clangle%20v_1%2C%20e_1%20%5Crangle%20%26%20%5Clangle%20v_2%2C%20e_1%20%5Crangle%20%26%20%5Ccdots%20%26%20%5Clangle%20v_m%2C%20e_1%20%5Crangle%20%5C%5C%20%0A0%20%26%20%5Clangle%20v_2%2C%20e_2%20%5Crangle%20%26%20%5Ccdots%20%26%20%5Clangle%20v_m%2C%20e_2%20%5Crangle%20%5C%5C%20%0A%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%5C%5C%20%0A0%20%26%200%20%26%20%5Ccdots%20%26%20%5Clangle%20v_m%2C%20e_m%20%5Crangle%20%5C%5C%0A%5Cend%7Bmatrix%7D%20%5Cright)%3D%20QR" alt="A=(v_1, v_2,..., v_m)=(e_1, e_2,..., e_m)\left( \begin{matrix} 
\langle v_1, e_1 \rangle &amp; \langle v_2, e_1 \rangle &amp; \cdots &amp; \langle v_m, e_1 \rangle \\ 
0 &amp; \langle v_2, e_2 \rangle &amp; \cdots &amp; \langle v_m, e_2 \rangle \\ 
\vdots &amp; \vdots &amp; \vdots &amp; \ddots\\ 
0 &amp; 0 &amp; \cdots &amp; \langle v_m, e_m \rangle \\
\end{matrix} \right)= QR" />  
với <img src="https://i.upmath.me/svg/Q" alt="Q" /> là ma trận có các cột trực giao, <img src="https://i.upmath.me/svg/R" alt="R" /> là ma trận tam giác trên vuông cấp <img src="https://i.upmath.me/svg/m" alt="m" /> 
- [Định lý]. Giả sử rằng <img src="https://i.upmath.me/svg/A%20%5Cin%20M_%7Bn%20%5Ctimes%20m%7D(%5Cmathbb%7BR%7D)" alt="A \in M_{n \times m}(\mathbb{R})" /> có các cột độc lập tuyến tính, khi đó ta có thể phân tích <img src="https://i.upmath.me/svg/A%3DQR" alt="A=QR" /> trong đó <img src="https://i.upmath.me/svg/Q" alt="Q" /> là ma trận trực giao và <img src="https://i.upmath.me/svg/R" alt="R" /> là ma trận tam giác trên cấp <img src="https://i.upmath.me/svg/m" alt="m" /> khả nghịch.  
Ví dụ: Cho ma trận  
<img src="https://i.upmath.me/svg/A%3D(v_1%2C%20v_2%2C%20v_3)%3D%5Cbegin%7Bpmatrix%7D%201%20%26%200%20%26%200%20%5C%5C%201%20%26%201%20%26%200%20%5C%5C%201%20%26%201%20%26%201%20%5Cend%7Bpmatrix%7D" alt="A=(v_1, v_2, v_3)=\begin{pmatrix} 1 &amp; 0 &amp; 0 \\ 1 &amp; 1 &amp; 0 \\ 1 &amp; 1 &amp; 1 \end{pmatrix}" /> 
[![qr1.png](https://i.postimg.cc/3JJ6ncdS/qr1.png)](https://postimg.cc/YjZbS8NY)
[![qr2.png](https://i.postimg.cc/VvHF74Vp/qr2.png)](https://postimg.cc/2Lhvydm7)


## 2. Code Python

```python
import numpy as np

#Chuẩn hóa vector
def normalize(vector):
    norm_vector=np.linalg.norm(vector)
    return vector/norm_vector

#Tìm ma trận trực giao Q bằng Gram - Schmidt Process
def GramSchmidtProcess(A):
    A=np.transpose(A)
    n=A.shape[1]
    B=np.zeros((n,n), dtype=np.float32)
    e_1=normalize(A[:,0])
    B[:,0]=e_1
    for i in range(1,n):
        temp=0
        for j in range(0,i):
            temp=temp+(np.inner(A[:,i],B[:,j]))*B[:,j]
        temp2=normalize(A[:,i]-temp)
        B[:,i]=temp2
    return B

#Cách 1: Tính R bằng ma trận nghịch đảo
#A=QR => R=Q^-1.dot(A)
def QR_Decomposition_GramSchimidt_1(A):
    Q=GramSchmidtProcess(A)
    R=(Q.T).dot(A.T)
    return Q,R

A=np.array([[1,1,1],[0,1,1],[0,0,1]])
Q,R=QR_Decomposition_GramSchimidt_1(A)
print(Q)
print(R)

#Cách 2
def QR_Decomposition_GramSchmidt_2(A):
    Q=GramSchmidtProcess(A)
    A=np.transpose(A)
    m=A.shape[0]
    R=np.zeros((m,m))
    for i in range(0, m):
        for j in range(0, i+1):
            R[j][i]=np.inner(A[:,i], Q[:,j])
    return Q,R
Q,R=QR_Decomposition_GramSchmidt_2(A)
print(Q)
print(R)
```

```python
A=np.array([[1,1,1],[0,1,1],[0,0,1]])
Q,R=QR_Decomposition_GramSchmidt_2(A)
print(Q)
print(R)
```

```python
[[ 5.7735026e-01 -8.1649655e-01  4.2146848e-08]
 [ 5.7735026e-01  4.0824834e-01 -7.0710677e-01]
 [ 5.7735026e-01  4.0824834e-01  7.0710677e-01]]
[[1.73205078 1.15470052 0.57735026]
 [0.         0.81649667 0.40824834]
 [0.         0.         0.70710677]]
```
Kết quả cho ra đúng với kết quả ta làm ở trên.  

# III. Tài liệu tham khảo
1. Slide bài giảng Đại số A2, Thầy Lê Văn Luyện, ĐH KHTN TPHCM.
2. https://fit.mta.edu.vn/files/DanhSach/BaigiangDSTT_16.pdf
