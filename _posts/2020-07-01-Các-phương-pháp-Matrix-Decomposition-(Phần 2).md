---
layout: post
title: Các phương pháp Matrix Decomposition (Phần 2)
tags: [Linear Algebra]
---  


- Bài trước mình đã nói về LU Decomposition, ở bài này mình sẽ trình bày về **Cholesky Decomposition**. Phân tích này được đề xuất bởi nhà toán học André-Louis Cholesky, nó có nhiều ứng dụng trong tính ma trận nghịch đảo, giải hệ phương trình tuyến tính, tính định thức, Least Squares Problem hay trong Non-linear Optimization để phân tích ma trận Hessian,...
- Trước hết chúng ta cùng xem lại một số khái niệm cơ bản.

# I. Các khái niệm cơ bản  

## 1. Ma trận chuyển vị và ma trận Hermitian
- Cho <img src="https://i.upmath.me/svg/A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bm%20%5Ctimes%20n%7D" alt="A \in \mathbb{R}^{m \times n}" />, ta nói <img src="https://i.upmath.me/svg/B%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20m%7D" alt="B \in \mathbb{R}^{n \times m}" /> là chuyển vị của <img src="https://i.upmath.me/svg/A" alt="A" /> nếu <img src="https://i.upmath.me/svg/b_%7Bij%7D%3D%20a_%7Bji%7D%2C%20%5Cforall%201%20%5Cle%20i%20%5Cle%20n%2C%201%20%5Cle%20j%20%5Cle%20m" alt="b_{ij}= a_{ji}, \forall 1 \le i \le n, 1 \le j \le m" />.  
- Nếu <img src="https://i.upmath.me/svg/A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bm%20%5Ctimes%20n%7D" alt="A \in \mathbb{R}^{m \times n}" /> thì <img src="https://i.upmath.me/svg/A%5ET%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20m%7D" alt="A^T \in \mathbb{R}^{n \times m}" />. Nếu <img src="https://i.upmath.me/svg/A%5ET%20%3DA" alt="A^T =A" />, ta nói <img src="https://i.upmath.me/svg/A" alt="A" /> là một ma trận đối xứng (symmetric matrix).  
- Cho <img src="https://i.upmath.me/svg/A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bm%20%5Ctimes%20n%7D" alt="A \in \mathbb{R}^{m \times n}" />, ta nói <img src="https://i.upmath.me/svg/B%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20m%7D" alt="B \in \mathbb{R}^{n \times m}" /> là chuyển vị liên hợp của <img src="https://i.upmath.me/svg/A" alt="A" /> nếu <img src="https://i.upmath.me/svg/b_%7Bij%7D%3D%20%5Coverline%7Ba_%7Bji%7D%7D%2C%20%5Cforall%201%20%5Cle%20i%20%5Cle%20n%2C%201%20%5Cle%20j%20%5Cle%20m" alt="b_{ij}= \overline{a_{ji}}, \forall 1 \le i \le n, 1 \le j \le m" />, trong đó <img src="https://i.upmath.me/svg/%5Coverline%7Ba%7D" alt="\overline{a}" /> là liên hợp phức của <img src="https://i.upmath.me/svg/A" alt="A" />. Nếu chuyển vị liên hợp của một ma trận phức <img src="https://i.upmath.me/svg/A%2C%20A%5Cin%20%5Cmathbb%7BC%7D%5E%7Bm%20%5Ctimes%20n%7D" alt="A, A\in \mathbb{C}^{m \times n}" /> bằng với chính nó, tức là <img src="https://i.upmath.me/svg/A%5EH%3DA" alt="A^H=A" />, thì ta nói ma trận đó là Hermitian.  

## 2. Ma trận xác định dương  
- Một ma trận vuông <img src="https://i.upmath.me/svg/A" alt="A" />, <img src="https://i.upmath.me/svg/A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20n%7D" alt="A \in \mathbb{R}^{n \times n}" /> được gọi là *xác định dương* (positive definite) nếu ma trận <img src="https://i.upmath.me/svg/A" alt="A" /> đối xứng (<img src="https://i.upmath.me/svg/A%5ET%3DA" alt="A^T=A" />) và thỏa mãn <img src="https://i.upmath.me/svg/x%5ETAx%20%3E0" alt="x^TAx &gt;0" /> với mọi <img src="https://i.upmath.me/svg/x%24%20%5Cne%200%2C%20x%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D" alt="x$ \ne 0, x \in \mathbb{R}^{n}" />. Nói cách khác <img src="https://i.upmath.me/svg/x%5ETAx%20%5Cge%200" alt="x^TAx \ge 0" /> với mọi <img src="https://i.upmath.me/svg/x" alt="x" /> và <img src="https://i.upmath.me/svg/x%5ETAx%3D0" alt="x^TAx=0" /> khi và chỉ khi <img src="https://i.upmath.me/svg/x%3D0" alt="x=0" />.  
- Một ma trận vuông <img src="https://i.upmath.me/svg/A" alt="A" />, <img src="https://i.upmath.me/svg/A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20n%7D" alt="A \in \mathbb{R}^{n \times n}" /> được gọi là ma trận *nửa xác định dương* (positive semidefinite) nếu ma trận đó đối xứng và thỏa mãn <img src="https://i.upmath.me/svg/x%5ETAx%20%5Cge%200%2C%20%5Cforall%20x%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D%20" alt="x^TAx \ge 0, \forall x \in \mathbb{R}^{n} " />.  
- Nhận xét: Mọi ma trận xác định dương cũng là ma trận nửa xác định dương nhưng điều ngược lại không đúng, bởi vì ma trận xác định dương có ràng buộc <img src="https://i.upmath.me/svg/x%5ETAx%3D0" alt="x^TAx=0" /> khi và chỉ khi <img src="https://i.upmath.me/svg/x%3D0" alt="x=0" />.  
- Với ma trận vuông <img src="https://i.upmath.me/svg/A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20n%7D" alt="A \in \mathbb{R}^{n \times n}" />, ta có <img src="https://i.upmath.me/svg/x%5ETAx%3D%20%5Cdisplaystyle%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Csum_%7Bj%3D1%7D%5E%7Bn%7DA_%7Bij%7Dx_ix_j" alt="x^TAx= \displaystyle \sum_{i=1}^{n} \sum_{j=1}^{n}A_{ij}x_ix_j" />, với <img src="https://i.upmath.me/svg/A" alt="A" /> đối xứng, ta có thể viết <img src="https://i.upmath.me/svg/x%5ETAx%3D%20%5Cdisplaystyle%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7DA_%7Bii%7Dx_i%5E2%2B%202%20%5Cdisplaystyle%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Csum_%7Bj%3D1%7D%5E%7Bi-1%7DA_%7Bij%7Dx_ix_j%20" alt="x^TAx= \displaystyle \sum_{i=1}^{n}A_{ii}x_i^2+ 2 \displaystyle \sum_{i=1}^{n} \sum_{j=1}^{i-1}A_{ij}x_ix_j " />.  
- **Ví dụ**: Với ma trận <img src="https://i.upmath.me/svg/A%3D%5Cbegin%7Bbmatrix%7D%209%20%26%206%20%5C%5C%206%26%205%20%5Cend%7Bbmatrix%7D" alt="A=\begin{bmatrix} 9 &amp; 6 \\ 6&amp; 5 \end{bmatrix}" /> ta có <img src="https://i.upmath.me/svg/x%5ETAx%3D%209x_1%5E2%2B12x_1x_2%2B5x_2%5E2%3D%20(3x_1%2B2x_2)%5E2%2B%20x_2%5E2%20%5Cge%200%2C%20%5Cforall%20x" alt="x^TAx= 9x_1^2+12x_1x_2+5x_2^2= (3x_1+2x_2)^2+ x_2^2 \ge 0, \forall x" />. Hơn nữa <img src="https://i.upmath.me/svg/x%5ETAx%3D0" alt="x^TAx=0" /> khi và chỉ khi <img src="https://i.upmath.me/svg/x_1%3Dx_2%3D0" alt="x_1=x_2=0" />. Do đó <img src="https://i.upmath.me/svg/A" alt="A" /> là ma trận xác định dương.  
- Dễ thấy các phần tử trên đường chéo chính của ma trận xác định dương là các số dương. Thật vậy với ma trận xác định dương <img src="https://i.upmath.me/svg/A" alt="A" />, xét <img src="https://i.upmath.me/svg/x%3De_i" alt="x=e_i" /> với <img src="https://i.upmath.me/svg/e_i" alt="e_i" /> là unit vector thứ <img src="https://i.upmath.me/svg/i" alt="i" />, ta có <img src="https://i.upmath.me/svg/x%5ETAx%3D%20e_i%5ETAe_i%3D%20A_%7Bii%7D%3E0" alt="x^TAx= e_i^TAe_i= A_{ii}&gt;0" /> với <img src="https://i.upmath.me/svg/i%3D1%2C2%2C...%2Cn" alt="i=1,2,...,n" />.  

## 3. Schur complement
- Cho <img src="https://i.upmath.me/svg/A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20n%7D" alt="A \in \mathbb{R}^{n \times n}" /> là ma trận xác định dương, viết lại <img src="https://i.upmath.me/svg/A" alt="A" /> như sau:  
<img src="https://i.upmath.me/svg/A%3D%5Cbegin%7Bbmatrix%7D%20A_%7B11%7D%20%26%20A_%7B2%3An%2C1%7D%5ET%20%5C%5C%20A_%7B2%3An%2C%201%7D%26%20A_%7B2%3An%2C%202%3An%7D%20%5Cend%7Bbmatrix%7D" alt="A=\begin{bmatrix} A_{11} &amp; A_{2:n,1}^T \\ A_{2:n, 1}&amp; A_{2:n, 2:n} \end{bmatrix}" />.  
Ma trận <img src="https://i.upmath.me/svg/S%3DA_%7B2%3An%2C2%3An%7D-%5Cdfrac%7B1%7D%7BA_%7B11%7D%7DA_%7B2%3An%2C1%7DA_%7B2%3An%2C1%7D%5ET" alt="S=A_{2:n,2:n}-\dfrac{1}{A_{11}}A_{2:n,1}A_{2:n,1}^T" /> gọi là Schur complement của <img src="https://i.upmath.me/svg/A_%7B11%7D" alt="A_{11}" />.  
- Bây giờ ta chứng minh <img src="https://i.upmath.me/svg/S" alt="S" /> cũng là ma trận xác định dương. Lấy vector <img src="https://i.upmath.me/svg/v%20%5Cne%200" alt="v \ne 0" />. Đặt <img src="https://i.upmath.me/svg/x%3D%20%5Cbegin%7Bbmatrix%7D%20-%5Cdfrac%7B1%7D%7BA_%7B11%7D%7DA_%7B2%3An%2C1%7D%5ETv%20%5C%5C%20v%20%5Cend%7Bbmatrix%7D" alt="x= \begin{bmatrix} -\dfrac{1}{A_{11}}A_{2:n,1}^Tv \\ v \end{bmatrix}" />. Ta có <img src="https://i.upmath.me/svg/x%20%5Cne%200" alt="x \ne 0" /> và  
<img src="https://i.upmath.me/svg/x%5ETAx%3D%5Cbegin%7Bbmatrix%7D%20-%5Cdfrac%7B1%7D%7BA_%7B11%7D%7DA_%7B2%3An%2C1%7D%5ETv%20%26%20v%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7D%20A_%7B11%7D%20%26%20A_%7B2%3An%2C1%7D%5ET%20%5C%5C%20A_%7B2%3An%2C%201%7D%26%20A_%7B2%3An%2C%202%3An%7D%20%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7D%20-%5Cdfrac%7B1%7D%7BA_%7B11%7D%7DA_%7B2%3An%2C1%7D%5ETv%20%5C%5C%20v%20%5Cend%7Bbmatrix%7D%20%3D" alt="x^TAx=\begin{bmatrix} -\dfrac{1}{A_{11}}A_{2:n,1}^Tv &amp; v\end{bmatrix}\begin{bmatrix} A_{11} &amp; A_{2:n,1}^T \\ A_{2:n, 1}&amp; A_{2:n, 2:n} \end{bmatrix}\begin{bmatrix} -\dfrac{1}{A_{11}}A_{2:n,1}^Tv \\ v \end{bmatrix} =" />   
<img src="https://i.upmath.me/svg/%5Cbegin%7Bbmatrix%7D%20-%5Cdfrac%7B1%7D%7BA_%7B11%7D%7DA_%7B2%3An%2C1%7D%5ETv%20%26%20v%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%200%20%5C%5CSv%20%20%5Cend%7Bbmatrix%7D%3D%20v%5ETSv%20" alt="\begin{bmatrix} -\dfrac{1}{A_{11}}A_{2:n,1}^Tv &amp; v\end{bmatrix} \begin{bmatrix} 0 \\Sv  \end{bmatrix}= v^TSv " />.  
Do <img src="https://i.upmath.me/svg/A" alt="A" /> là ma trận xác định dương nên <img src="https://i.upmath.me/svg/x%5ETAx%3E0" alt="x^TAx&gt;0" /> do đó <img src="https://i.upmath.me/svg/v%5ETSv%20%3E0" alt="v^TSv &gt;0" />, vậy <img src="https://i.upmath.me/svg/S" alt="S" /> cũng là ma trận xác định dương.  

## 4. Gram matrix  
- Ma trận Gram là ma trận có dạng <img src="https://i.upmath.me/svg/B%5ETB" alt="B^TB" />.
- Ma trận Gram thì nửa xác định dương vì <img src="https://i.upmath.me/svg/x%5ET%20B%5ETBx%20%3D%20(Bx)%5ET(Bx)%20%3D%7C%7CBx%7C%7C%5E2%20%5Cge%200" alt="x^T B^TBx = (Bx)^T(Bx) =||Bx||^2 \ge 0" /> với mọi <img src="https://i.upmath.me/svg/x" alt="x" />.  
- Ma trận Gram là xác định dương khi <img src="https://i.upmath.me/svg/%7C%7CBx%7C%7C%3D0%20%5Cimplies%20x%3D0" alt="||Bx||=0 \implies x=0" /> hay các cột của <img src="https://i.upmath.me/svg/B" alt="B" /> là độc lập tuyến tính.  

# II. Cholesky Decomposition
## 1. Định nghĩa
- Mọi ma trận xác định dương <img src="https://i.upmath.me/svg/A" alt="A" /> đều có thể được phân tích thành <img src="https://i.upmath.me/svg/A%3DR%5ETR" alt="A=R^TR" /> trong đó <img src="https://i.upmath.me/svg/R" alt="R" /> là ma trận tam giác trên (hoặc <img src="https://i.upmath.me/svg/A%3DRR%5ET" alt="A=RR^T" /> với <img src="https://i.upmath.me/svg/R" alt="R" /> là ma trận tam giác dưới) với các phần tử trên đường chéo chính là các số thực dương.  
- Ví dụ với ma trận xác định dương <img src="https://i.upmath.me/svg/%203%20%5Ctimes%203" alt=" 3 \times 3" />  
<img src="https://i.upmath.me/svg/%5Cbegin%7Bbmatrix%7D%2025%20%26%2015%20%26%20-5%20%5C%5C%2015%20%26%2018%20%26%200%20%5C%5C%20-5%20%26%200%20%26%2011%20%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D%205%20%26%200%20%26%200%20%5C%5C%203%20%26%203%20%26%200%20%5C%5C%20-1%20%26%201%20%26%203%20%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7D%205%20%26%203%20%26%20-1%20%5C%5C%200%20%26%203%20%26%201%20%5C%5C%200%20%26%200%20%26%203%20%5Cend%7Bbmatrix%7D" alt="\begin{bmatrix} 25 &amp; 15 &amp; -5 \\ 15 &amp; 18 &amp; 0 \\ -5 &amp; 0 &amp; 11 \end{bmatrix}=\begin{bmatrix} 5 &amp; 0 &amp; 0 \\ 3 &amp; 3 &amp; 0 \\ -1 &amp; 1 &amp; 3 \end{bmatrix}\begin{bmatrix} 5 &amp; 3 &amp; -1 \\ 0 &amp; 3 &amp; 1 \\ 0 &amp; 0 &amp; 3 \end{bmatrix}" />  

## 2. Thuật toán tìm Cholesky Decomposition
- Ta có thể viết lại <img src="https://i.upmath.me/svg/A%3DR%5ETR%20" alt="A=R^TR " /> như sau:  
<img src="https://i.upmath.me/svg/A%3D%5Cbegin%7Bbmatrix%7D%20A_%7B11%7D%20%26%20A_%7B1%2C%202%3An%7D%20%5C%5C%20A_%7B2%3An%2C%201%7D%26%20A_%7B2%3An%2C%202%3An%7D%20%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D%20R_%7B11%7D%20%26%200%20%5C%5C%20R_%7B1%2C%202%3An%7D%5ET%26%20R_%7B2%3An%2C%202%3An%7D%5ET%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7D%20R_%7B11%7D%20%26%20R_%7B1%2C%202%3An%7D%20%5C%5C%200%26%20R_%7B2%3An%2C%202%3An%7D%5Cend%7Bbmatrix%7D%3D%20" alt="A=\begin{bmatrix} A_{11} &amp; A_{1, 2:n} \\ A_{2:n, 1}&amp; A_{2:n, 2:n} \end{bmatrix}=\begin{bmatrix} R_{11} &amp; 0 \\ R_{1, 2:n}^T&amp; R_{2:n, 2:n}^T\end{bmatrix}\begin{bmatrix} R_{11} &amp; R_{1, 2:n} \\ 0&amp; R_{2:n, 2:n}\end{bmatrix}= " />  
<img src="https://i.upmath.me/svg/%5Cbegin%7Bbmatrix%7D%20R_%7B11%7D%5E2%20%26%20R_%7B11%7DR_%7B1%2C%202%3An%7D%20%5C%5C%20R_%7B11%7DR_%7B1%2C%202%3An%7D%5ET%20%26%20R_%7B1%2C%202%3An%7D%5ET%20R_%7B1%2C%202%3An%7D%20%2B%20R_%7B2%3An%2C%202%3An%7D%5ETR_%7B2%3An%2C%202%3An%7D%20%5Cend%7Bbmatrix%7D" alt="\begin{bmatrix} R_{11}^2 &amp; R_{11}R_{1, 2:n} \\ R_{11}R_{1, 2:n}^T &amp; R_{1, 2:n}^T R_{1, 2:n} + R_{2:n, 2:n}^TR_{2:n, 2:n} \end{bmatrix}" />.  
Từ đây ta suy <img src="https://i.upmath.me/svg/R_%7B11%7D%3D%5Csqrt%7BA_%7B11%7D%7D" alt="R_{11}=\sqrt{A_{11}}" />, <img src="https://i.upmath.me/svg/R_%7B1%2C2%3An%7D%3D%20%5Cdfrac%7B1%7D%7BR_%7B11%7D%7DA_%7B1%2C2%3An%7D" alt="R_{1,2:n}= \dfrac{1}{R_{11}}A_{1,2:n}" />,  
<img src="https://i.upmath.me/svg/R_%7B2%3An%2C2%3An%7D%5ETR_%7B2%3An%2C2%3An%7D%3DA_%7B2%3An%2C%202%3An%7D-R_%7B1%2C%202%3An%7D%5ET%20R_%7B1%2C%202%3An%7D" alt="R_{2:n,2:n}^TR_{2:n,2:n}=A_{2:n, 2:n}-R_{1, 2:n}^T R_{1, 2:n}" />.  
Do đó <img src="https://i.upmath.me/svg/R_%7B2%3An%2C2%3An%7D%5ETR_%7B2%3An%2C2%3An%7D" alt="R_{2:n,2:n}^TR_{2:n,2:n}" /> là phân tích Cholesky của <img src="https://i.upmath.me/svg/A_%7B2%3An%2C%202%3An%7D-R_%7B1%2C%202%3An%7D%5ET%20R_%7B1%2C%202%3An%7D%3D%20A_%7B2%3An%2C%202%3An%7D-%5Cdfrac%7B1%7D%7BA_%7B11%7D%7DA_%7B2%3An%2C1%7DA_%7B2%3An%2C1%7D%5ET" alt="A_{2:n, 2:n}-R_{1, 2:n}^T R_{1, 2:n}= A_{2:n, 2:n}-\dfrac{1}{A_{11}}A_{2:n,1}A_{2:n,1}^T" />.  
Để ý thấy biểu thức cuối là Schur complement của <img src="https://i.upmath.me/svg/A_%7B11%7D" alt="A_{11}" />. Theo chứng minh ở phần I.3 thì đây là ma trận xác định dương kích thước <img src="https://i.upmath.me/svg/(n-1)%20%5Ctimes%20(n-1)" alt="(n-1) \times (n-1)" />. Cứ tiếp tục truy hồi đến phân tích Cholesky <img src="https://i.upmath.me/svg/%201%20%5Ctimes%201" alt=" 1 \times 1" /> ta được kết quả. 
[![105672907-965851400501778-5993859188482975687-n.png](https://i.postimg.cc/DZkBbD2Q/105672907-965851400501778-5993859188482975687-n.png)](https://postimg.cc/7C9gFKJb) 
**Ví dụ**: Tìm phân tích Cholesky của ma trận  
<img src="https://i.upmath.me/svg/%5Cbegin%7Bbmatrix%7D%2025%20%26%2015%20%26%20-5%20%5C%5C%2015%20%26%2018%20%26%200%20%5C%5C%20-5%20%26%200%20%26%2011%20%5Cend%7Bbmatrix%7D" alt="\begin{bmatrix} 25 &amp; 15 &amp; -5 \\ 15 &amp; 18 &amp; 0 \\ -5 &amp; 0 &amp; 11 \end{bmatrix}" />  
[![106213020-2692502164353723-4322951759308171152-n.png](https://i.postimg.cc/7YVdrxqF/106213020-2692502164353723-4322951759308171152-n.png)](https://postimg.cc/N5Kp1v9D)
Để lập trình được, các phần tử trong ma trận được viết lại thành  
[![106234882-2609404785965630-4127219280057773479-n.png](https://i.postimg.cc/br3vt7js/106234882-2609404785965630-4127219280057773479-n.png)](https://postimg.cc/PvvhGFJj)  
- Code Python
```python
import numpy as np
def CholeskyDecomposition(A):
    n=A.shape[0]
    L=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            if(i==j):
                temp=A[i,i]-np.sum(L[i,:i]*L[i,:i])
                if temp<0.0:
                    return 0.0
                L[i][i]=np.sqrt(temp)
            else:
                L[i][j]=(A[i][j]-np.sum(L[i,:j]*L[j,:j]))/(L[j][j])
    return L.T
```  
Kết quả chạy  

```python
A=np.array([[25,15,-5],[15,18,0],[-5,0,11]])
R=CholeskyDecomposition(A)
print("R=",R)
print("Check")
print("A= ", R.T @ R)
```  
  
  
```python
R= [[ 5.  3. -1.]
 [ 0.  3.  1.]
 [ 0.  0.  3.]]
Check
A=  [[25. 15. -5.]
 [15. 18.  0.]
 [-5.  0. 11.]]
```  
  
Kết quả cho ra đúng với kết quả ta làm ở trên.  

- Ta có thể dùng hàm cholesky trong thư viện <img src="https://i.upmath.me/svg/np.linalg" alt="np.linalg" />.  

Code Python:  

```python
R=np.linalg.cholesky(A)
print(L)
R_T=L.transpose()
print(R.dot(R_T))
```  

Kết quả hoàn toàn giống.  

# III. Ứng dụng Cholesky Decomposition  

## 1. Giải hệ phương trình tuyến tính
Giả sử hệ phương trình tuyến tính được biểu diễn dưới dạng ma trận <img src="https://i.upmath.me/svg/Ax%3Db" alt="Ax=b" /> trong đó <img src="https://i.upmath.me/svg/A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20n%7D" alt="A \in \mathbb{R}^{n \times n}" /> là ma trận xác định dương, <img src="https://i.upmath.me/svg/x%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D%2C%20b%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D%20" alt="x \in \mathbb{R}^{n}, b\in \mathbb{R}^{n} " />.  
<img src="https://i.upmath.me/svg/Ax%3Db" alt="Ax=b" />  
<img src="https://i.upmath.me/svg/RR%5ETx%3Db" alt="RR^Tx=b" />  
<img src="https://i.upmath.me/svg/R%5ETx%3D%20R%5E%7B-1%7Db" alt="R^Tx= R^{-1}b" />  
<img src="https://i.upmath.me/svg/x%3D(R%5ET)%5E%7B-1%7D(R%5E%7B-1%7Db)" alt="x=(R^T)^{-1}(R^{-1}b)" />
- Code Python
```python
def CholeskyLinearEquation(A,b):
    #Find x with Ax=b
    L=np.linalg.cholesky(A)
    y=(np.linalg.inv(L)).dot(b)
    x=(np.linalg.inv(np.transpose(L))).dot(y)
    return x
```
```python
A=np.array([[4,12,-16],[12,37,-43],[-16,-43,98]])
b=np.array([ -68 ,-191 , 364])
x=CholeskyLinearEquation(A,b)
print("x=",x)
```
Kết quả chạy: Hệ có nghiệm  
```python
x= [ 1. -2.  3.]
```  

## 2. Least Squares Problem trong Linear Regression
Tham khảo bài viết tại <https://alexisalulema.com/2018/01/20/cholesky-decomposition-for-linear-regression-with-tensorflow/>  

# IV. Cholesky Decomposition của ma trận Gram
- Cho ma trận <img src="https://i.upmath.me/svg/B%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bm%20%5Ctimes%20n%7D" alt="B \in \mathbb{R}^{m \times n}" /> có các cột độc lập tuyến tính.  
Khi đó ma trận Gram  <img src="https://i.upmath.me/svg/A%3D%20B%5ETB" alt="A= B^TB" /> là ma trận xác định dương. Có 2 cách để tính Cholesky Decomposition của <img src="https://i.upmath.me/svg/A" alt="A" />.  
1. Tính trực tiếp <img src="https://i.upmath.me/svg/A%3DR%5ETR" alt="A=R^TR" /> theo phương pháp ở trên.  
2. Tính **QR Decomposition** của <img src="https://i.upmath.me/svg/B" alt="B" />, tức là <img src="https://i.upmath.me/svg/B%3DQR" alt="B=QR" />, khi đó <img src="https://i.upmath.me/svg/A%3DB%5ETB%3DR%5ETQ%5ETQR%3DR%5ETR" alt="A=B^TB=R^TQ^TQR=R^TR" />, <img src="https://i.upmath.me/svg/R" alt="R" /> cũng là ma trận trong cách tính <img src="https://i.upmath.me/svg/1" alt="1" />.  
Về phân tích QR mình sẽ trình bày ở bài sau.  


# III. Tài liệu tham khảo
1. <http://www.seas.ucla.edu/~vandenbe/133A/133A-notes.pdf>
2. <https://alexisalulema.com/2018/01/20/cholesky-decomposition-for-linear-regression-with-tensorflow/>
3. Ebook Machine Learning cơ bản, Vũ Hữu Tiệp.
4. <https://www.geeksforgeeks.org/cholesky-decomposition-matrix-decomposition/>
5. <http://www.seas.ucla.edu/~vandenbe/133A/lectures/chol.pdf>
























