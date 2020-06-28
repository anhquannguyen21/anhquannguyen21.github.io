---
layout: post
title: Các phương pháp Matrix Decomposition (Phần 1)
tags: [Linear Algebra]
---


# I. Giới thiệu  
- Trong Machine Learning, chúng ta thường xử lý dữ liệu được biểu diễn dưới dạng ma trận, thường các ma trận này có kích thước lớn. Rất nhiều các bài toán học máy được giải quyết bằng cách sử dụng các phương pháp của Đại số tuyến tính. Trong các bài này mình sẽ trình bày về các phương pháp phân tích ma trận (hay Phân rã ma trận) (Matrix Decomposition).  
- Việc phân tích một ma trận là đưa ma trận đó về tích của 2 hay nhiều ma trận đặc biệt khác, thường là ma trận đường chéo và ma trận tam giác. Việc phân tích này nhằm mục đích dễ dàng tính định thức, tìm ma trận nghịch đảo, giải hệ phương trình tuyến tính, giảm chiều dữ liệu,... Matrix Decomposition cũng được ứng dụng trong bài toán về Hệ thống khuyến nghị (Recommendation System).  
- Một số phương pháp phân tích ma trận phổ biến như LU, QR, Cholesky, Eigen Decomposition (Chéo hóa ma trận), SVD,... Trong đó, SVD được sử dụng nhiều trong các thuật toán Học máy và Thị giác máy tính. Trong phần này mình sẽ trình bày một phương pháp đơn giản nhất là phân tích LU. (LU là viết tắt của Lower Triangular Matrix và Upper Triangular Matrix)  
# II. LU Decomposition
## 1. Định nghĩa
- Cho <img src="https://i.upmath.me/svg/A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20n%7D" alt="A \in \mathbb{R}^{n \times n}" /> là một ma trận vuông, phân tích LU của <img src="https://i.upmath.me/svg/A" alt="A" /> là cách viết <img src="https://i.upmath.me/svg/A" alt="A" /> thành tích của 2 ma trận có dạng <img src="https://i.upmath.me/svg/A%3DLU" alt="A=LU" /> trong đó <img src="https://i.upmath.me/svg/L%2CU" alt="L,U" /> là các ma trận tam giác dưới và tam giác trên có cùng kích thước với <img src="https://i.upmath.me/svg/A" alt="A" />. Người ta chứng minh được rằng mọi ma trận cấp <img src="https://i.upmath.me/svg/n" alt="n" /> thỏa mãn phân tích <img src="https://i.upmath.me/svg/LU" alt="LU" /> khi và chỉ khi các định thức con chính khác <img src="https://i.upmath.me/svg/0" alt="0" />, phân tích sẽ *duy nhất* nếu thêm điều kiện các phần tử trên đường chéo chính của ma trận <img src="https://i.upmath.me/svg/L" alt="L" /> (hoặc <img src="https://i.upmath.me/svg/U" alt="U" />) đều bằng <img src="https://i.upmath.me/svg/1" alt="1" />.  
- Ví dụ với ma trận vuông <img src="https://i.upmath.me/svg/%203%20%5Ctimes%203" alt=" 3 \times 3" />  
<img src="https://i.upmath.me/svg/A%3D%5Cbegin%7Bbmatrix%7D%20a_%7B11%7D%20%26%20a_%7B12%7D%20%26%20a_%7B13%7D%20%5C%5C%20a_%7B21%7D%20%26%20a_%7B22%7D%20%26%20a_%7B23%7D%20%5C%5C%20a_%7B31%7D%20%26%20a_%7B32%7D%20%20%26%20a_%7B33%7D%20%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%200%20%5C%5C%20l_%7B21%7D%20%26%201%20%26%200%20%5C%5C%20l_%7B31%7D%20%26%20l_%7B32%7D%20%20%26%201%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20u_%7B11%7D%20%26%20u_%7B12%7D%20%26%20u_%7B13%7D%20%5C%5C%200%20%26%20u_%7B22%7D%20%26%20u_%7B23%7D%20%5C%5C%200%20%26%200%20%20%26%20a_%7B33%7D%20%5Cend%7Bbmatrix%7D%20" alt="A=\begin{bmatrix} a_{11} &amp; a_{12} &amp; a_{13} \\ a_{21} &amp; a_{22} &amp; a_{23} \\ a_{31} &amp; a_{32}  &amp; a_{33} \end{bmatrix}=\begin{bmatrix} 1 &amp; 0 &amp; 0 \\ l_{21} &amp; 1 &amp; 0 \\ l_{31} &amp; l_{32}  &amp; 1 \end{bmatrix} \begin{bmatrix} u_{11} &amp; u_{12} &amp; u_{13} \\ 0 &amp; u_{22} &amp; u_{23} \\ 0 &amp; 0  &amp; a_{33} \end{bmatrix} " />  
- Vể mặt toán học, tìm phân tích này ta làm như sau:  
*Bước 1*: Biến đổi sơ cấp ma trận <img src="https://i.upmath.me/svg/A" alt="A" />  thành ma trận tam giác trên <img src="https://i.upmath.me/svg/U" alt="U" />. Bản chất của quá trình này là nhân <img src="https://i.upmath.me/svg/A" alt="A" /> với dãy ma trận không suy biến dạng tam giác dưới, giả sử dãy đó là <img src="https://i.upmath.me/svg/C%3D%20C_1C_2...C_k" alt="C= C_1C_2...C_k" />, ta có <img src="https://i.upmath.me/svg/U%3DC_1C_2...C_kA" alt="U=C_1C_2...C_kA" />  
*Bước 2*: Do <img src="https://i.upmath.me/svg/LU%3DA" alt="LU=A" /> nên tìm được <img src="https://i.upmath.me/svg/L" alt="L" /> bằng công thức  
<img src="https://i.upmath.me/svg/L%3DC_k%5E%7B-1%7DC_%7Bk-1%7D%5E%7B-1%7D...C_1%5E%7B-1%7D%3DC%5E%7B-1%7D" alt="L=C_k^{-1}C_{k-1}^{-1}...C_1^{-1}=C^{-1}" />  
**Ví dụ**: Phân tích <img src="https://i.upmath.me/svg/LU" alt="LU" /> ma trận 
  
<img src="https://i.upmath.me/svg/A%3D%5Cbegin%7Bbmatrix%7D%206%20%26%2018%20%26%203%20%5C%5C%202%20%26%2012%20%26%201%20%5C%5C%204%20%26%2015%20%20%26%203%20%5Cend%7Bbmatrix%7D" alt="A=\begin{bmatrix} 6 &amp; 18 &amp; 3 \\ 2 &amp; 12 &amp; 1 \\ 4 &amp; 15  &amp; 3 \end{bmatrix}" />
  
[![106179321-303873930784112-5374058648779268154-n.png](https://i.postimg.cc/d1TnrJwB/106179321-303873930784112-5374058648779268154-n.png)](https://postimg.cc/62tdNJ72)  
[![105683156-1189060454801575-774184744179493534-n.png](https://i.postimg.cc/SRk8R4sj/105683156-1189060454801575-774184744179493534-n.png)](https://postimg.cc/Q9YVYRcr)
## 2. Thuật toán tìm LU Decomposition 
- Ta có thể biểu diễn <img src="https://i.upmath.me/svg/A%3DLU" alt="A=LU" /> như sau:  
<img src="https://i.upmath.me/svg/%5Cbegin%7Bbmatrix%7D%20A_%7B11%7D%20%26%20A_%7B1%2C%202%3An%7D%20%5C%5CA_%7B2%3An%2C1%7D%20%26%20A_%7B2%3An%2C2%3An%7D%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%5C%5CL_%7B2%3An%2C1%7D%20%26%20L_%7B2%3An%2C2%3An%7D%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7D%20U_%7B11%7D%20%26%20U_%7B1%2C%202%3An%7D%20%5C%5C0%20%26%20U_%7B2%3An%2C2%3An%7D%5Cend%7Bbmatrix%7D" alt="\begin{bmatrix} A_{11} &amp; A_{1, 2:n} \\A_{2:n,1} &amp; A_{2:n,2:n}\end{bmatrix}= \begin{bmatrix} 1 &amp; 0 \\L_{2:n,1} &amp; L_{2:n,2:n}\end{bmatrix}\begin{bmatrix} U_{11} &amp; U_{1, 2:n} \\0 &amp; U_{2:n,2:n}\end{bmatrix}" />  
(Kí hiệu <img src="https://i.upmath.me/svg/A_%7Bx_1%3Ax_2%2C%20y_1%3Ay_2%7D" alt="A_{x_1:x_2, y_1:y_2}" /> biểu diễn ma trận con lấy từ ma trận <img src="https://i.upmath.me/svg/A" alt="A" /> với hàng từ <img src="https://i.upmath.me/svg/x_1" alt="x_1" /> đến <img src="https://i.upmath.me/svg/x_2" alt="x_2" /> và các cột từ <img src="https://i.upmath.me/svg/y_1" alt="y_1" /> đến <img src="https://i.upmath.me/svg/y_2" alt="y_2" />)  
Suy ra <img src="https://i.upmath.me/svg/%5Cbegin%7Bbmatrix%7D%20A_%7B11%7D%20%26%20A_%7B1%2C%202%3An%7D%20%5C%5CA_%7B2%3An%2C1%7D%20%26%20A_%7B2%3An%2C2%3An%7D%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20U_%7B11%7D%20%26%20U_%7B1%2C%202%3An%7D%20%5C%5CU_%7B11%7DL_%7B2%3An%2C1%7D%20%26%20L_%7B2%3An%2C1%7DU_%7B1%2C2%3An%7D%20%2B%20L_%7B2%3An%2C%202%3An%7DU_%7B2%3An%2C%202%3An%7D%5Cend%7Bbmatrix%7D" alt="\begin{bmatrix} A_{11} &amp; A_{1, 2:n} \\A_{2:n,1} &amp; A_{2:n,2:n}\end{bmatrix} = \begin{bmatrix} U_{11} &amp; U_{1, 2:n} \\U_{11}L_{2:n,1} &amp; L_{2:n,1}U_{1,2:n} + L_{2:n, 2:n}U_{2:n, 2:n}\end{bmatrix}" />  
Từ đây ta có  
<img src="https://i.upmath.me/svg/U_%7B11%7D%3DA_%7B11%7D%2C%20U_%7B1%2C2%3An%7D%3DA_%7B1%2C2%3An%7D%2C%20L_%7B2%3An%2C1%7D%3D%5Cdfrac%7B1%7D%7BA_%7B11%7D%7DA_%7B2%3An%2C1%7D" alt="U_{11}=A_{11}, U_{1,2:n}=A_{1,2:n}, L_{2:n,1}=\dfrac{1}{A_{11}}A_{2:n,1}" />  
<img src="https://i.upmath.me/svg/L_%7B2%3An%2C2%3An%7DU_%7B2%3An%2C%202%3An%7D%3DA_%7B2%3An%2C%202%3An%7D-L_%7B2%3An%2C1%7DU_%7B1%2C2%3An%7D%3D%20%5Cdfrac%7B1%7D%7BA_%7B11%7D%7DA_%7B2%3An%2C1%7DA_%7B1%2C2%3An%7D" alt="L_{2:n,2:n}U_{2:n, 2:n}=A_{2:n, 2:n}-L_{2:n,1}U_{1,2:n}= \dfrac{1}{A_{11}}A_{2:n,1}A_{1,2:n}" />  
Đặt <img src="https://i.upmath.me/svg/S_%7B22%7D%3D%5Cdfrac%7B1%7D%7BA_%7B11%7D%7DA_%7B2%3An%2C1%7DA_%7B1%2C2%3An%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B(n-1)%20%5Ctimes%20(n-1)%7D" alt="S_{22}=\dfrac{1}{A_{11}}A_{2:n,1}A_{1,2:n} \in \mathbb{R}^{(n-1) \times (n-1)}" /> ta có  
<img src="https://i.upmath.me/svg/S_%7B22%7D%3DL_%7B2%3An%2C2%3An%7DU_%7B2%3An%2C%202%3An%7D" alt="S_{22}=L_{2:n,2:n}U_{2:n, 2:n}" /> là phân tích <img src="https://i.upmath.me/svg/LU" alt="LU" /> của <img src="https://i.upmath.me/svg/S_%7B22%7D" alt="S_{22}" />. 
(<img src="https://i.upmath.me/svg/S_%7B22%7D" alt="S_{22}" /> được gọi là **Schur complement**)  
Thuật toán tìm phân tích <img src="https://i.upmath.me/svg/LU" alt="LU" /> được mô tả bởi:  
[![104871302-213517126406628-5781667955209801506-n.png](https://i.postimg.cc/ZnPD1J9c/104871302-213517126406628-5781667955209801506-n.png)](https://postimg.cc/gLJDhFHX)
**Ví dụ**: Tìm phân tích <img src="https://i.upmath.me/svg/LU" alt="LU" /> của  

<img src="https://i.upmath.me/svg/A%3D%5Cbegin%7Bbmatrix%7D%208%20%26%202%20%26%209%20%5C%5C%204%26%209%20%26%204%20%5C%5C%206%20%26%207%20%20%26%209%20%5Cend%7Bbmatrix%7D" alt="A=\begin{bmatrix} 8 &amp; 2 &amp; 9 \\ 4&amp; 9 &amp; 4 \\ 6 &amp; 7  &amp; 9 \end{bmatrix}" />  

Ta có
<img src="https://i.upmath.me/svg/A%3D%5Cbegin%7Bbmatrix%7D%208%20%26%202%20%26%209%20%5C%5C%204%26%209%20%26%204%20%5C%5C%206%20%26%207%20%20%26%209%20%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%200%20%5C%5C%20L_%7B21%7D%20%26%201%20%26%200%20%5C%5C%20L_%7B31%7D%20%26%20L_%7B32%7D%20%20%26%201%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20U_%7B11%7D%20%26%20U_%7B12%7D%20%26%20U_%7B13%7D%20%5C%5C%200%20%26%20U_%7B22%7D%20%26%20U_%7B23%7D%20%5C%5C%200%20%26%200%20%20%26%20U_%7B33%7D%20%5Cend%7Bbmatrix%7D%20" alt="A=\begin{bmatrix} 8 &amp; 2 &amp; 9 \\ 4&amp; 9 &amp; 4 \\ 6 &amp; 7  &amp; 9 \end{bmatrix}=\begin{bmatrix} 1 &amp; 0 &amp; 0 \\ L_{21} &amp; 1 &amp; 0 \\ L_{31} &amp; L_{32}  &amp; 1 \end{bmatrix} \begin{bmatrix} U_{11} &amp; U_{12} &amp; U_{13} \\ 0 &amp; U_{22} &amp; U_{23} \\ 0 &amp; 0  &amp; U_{33} \end{bmatrix} " />  

Áp dụng các công thức ở trên ta được
[![105526869-947850485638003-1387427616597315388-n.png](https://i.postimg.cc/BnZQ768m/105526869-947850485638003-1387427616597315388-n.png)](https://postimg.cc/zVMrVqFh)  
Thuật toán ở trên có thể thể hiện mã giả (peusedocode) như sau  
[![106418939-713758882807182-5415356513796754688-n.png](https://i.postimg.cc/0jCWZmyf/106418939-713758882807182-5415356513796754688-n.png)](https://postimg.cc/hQf1tXxz)  
- Code Python  

```python
import numpy as np
def LU_Decomposition(A):
  n=A.shape[0]
  U=A.copy()
  L=np.zeros((n,n))
  for i in range(n):
    L[i,i]=1
  for k in range(n-1):
    for j in range(k+1, n):
      L[j,k]=U[j,k]/U[k,k]
      U[j, k:n]=U[j,k:n]-L[j,k]*U[k, k:n]
  return L,U
```
- Kết quả chạy thử:  
```python
A=np.array([[8.,2.,9.],[4.,9.,4.],[6.,7.,9.]])
print(A)
L,U=LU_Decomposition(A)
print("L=",L)
print("U=",U)
```
```python
L= [[1.     0.     0.    ]
 [0.5    1.     0.    ]
 [0.75   0.6875 1.    ]]
U= [[ 8.       2.       9.     ]
 [ 0.       8.      -0.5    ]
 [ 0.       0.       2.59375]]
```
Kết quả cho ra đúng với ta làm ở trên.
- Một cài đặt khác của thuật toán trên 
```python
import numpy as np
def LU_Decomposition(A):
    n= A.shape[0]
    L=np.zeros((n,n))
    U=np.zeros((n,n))
    M=A.copy()
    for i in range(n):
      U[i,i:]=M[i,i:]
      L[i:,i]=M[i:,i]/U[i,i]
      M[i+1:, i+1:]-=np.outer(L[i+1:,i], U[i, i+1:])
    return L,U
```
Kết quả cho ra hoàn toàn tương tự.  

# III. LU Decomposition with pivoting 
- Tổng quát hơn với ma trận vuông khả nghịch <img src="https://i.upmath.me/svg/A" alt="A" /> ta có phân tích <img src="https://i.upmath.me/svg/LUP" alt="LUP" />, đó là phân tích có dạng <img src="https://i.upmath.me/svg/PA%3DLU" alt="PA=LU" /> trong đó <img src="https://i.upmath.me/svg/L%2C%20U" alt="L, U" /> là các ma trận tam giác như trên, <img src="https://i.upmath.me/svg/P" alt="P" /> là ma trận nhận được trong biến đổi sơ cấp hàng, hay được gọi là ma trận hoán vị (permutation matrix).  
- Ma trận <img src="https://i.upmath.me/svg/P" alt="P" /> chỉ có các phần tử <img src="https://i.upmath.me/svg/0%2C1" alt="0,1" />. Ví dụ với ma trận <img src="https://i.upmath.me/svg/%203%20%5Ctimes%203" alt=" 3 \times 3" />  
<img src="https://i.upmath.me/svg/A%3D%5Cbegin%7Bbmatrix%7D%200%20%26%205%20%26%205%20%5C%5C%202%20%26%209%20%26%200%20%5C%5C%206%20%26%208%20%20%26%208%20%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D%200%20%26%200%20%26%201%20%5C%5C%200%20%26%201%20%26%200%20%5C%5C%201%20%26%200%20%20%26%200%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%200%20%5C%5C%20%5Cdfrac%7B1%7D%7B3%7D%20%26%201%20%26%200%20%5C%5C%200%20%26%20%5Cdfrac%7B15%7D%7B19%7D%20%20%26%201%20%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7D%206%20%26%208%20%26%208%20%5C%5C%200%20%26%20%5Cdfrac%7B19%7D%7B3%7D%20%26%20%5Cdfrac%7B-8%7D%7B3%7D%5C%5C%200%20%26%200%20%26%20%5Cdfrac%7B135%7D%7B19%7D%20%5Cend%7Bbmatrix%7D%20" alt="A=\begin{bmatrix} 0 &amp; 5 &amp; 5 \\ 2 &amp; 9 &amp; 0 \\ 6 &amp; 8  &amp; 8 \end{bmatrix}=\begin{bmatrix} 0 &amp; 0 &amp; 1 \\ 0 &amp; 1 &amp; 0 \\ 1 &amp; 0  &amp; 0 \end{bmatrix} \begin{bmatrix} 1 &amp; 0 &amp; 0 \\ \dfrac{1}{3} &amp; 1 &amp; 0 \\ 0 &amp; \dfrac{15}{19}  &amp; 1 \end{bmatrix}\begin{bmatrix} 6 &amp; 8 &amp; 8 \\ 0 &amp; \dfrac{19}{3} &amp; \dfrac{-8}{3}\\ 0 &amp; 0 &amp; \dfrac{135}{19} \end{bmatrix} " />  
- Ta có thể sử dụng hàm <img src="https://i.upmath.me/svg/lu" alt="lu" /> trong thư viện <img src="https://i.upmath.me/svg/scipy.linalg" alt="scipy.linalg" />  

```python
import numpy as np
from scipy.linalg import lu

A=np.array([[0,5,5],[2,3,0],[6,9,8]])
P,L,U=lu(A)
print("P=",P.T) #Trong hàm lu cho kết quả của A=PLU
print("L=", L)
print("U=", U)
print(P.T @ A)
print(L @ U)
```
Kết quả chạy

```python
P= [[0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]
L= [[1.         0.         0.        ]
 [0.         1.         0.        ]
 [0.33333333 0.         1.        ]]
U= [[ 6.          9.          8.        ]
 [ 0.          5.          5.        ]
 [ 0.          0.         -2.66666667]]
[[6. 9. 8.]
 [0. 5. 5.]
 [2. 3. 0.]]
[[6. 9. 8.]
 [0. 5. 5.]
 [2. 3. 0.]]
```
# III. Ứng dụng LU Decomposition
# 1. Giải hệ phương trình tuyến tính
Cho hệ phương phương trình tuyến tính biểu diễn dưới dạng ma trận <img src="https://i.upmath.me/svg/Ax%3Db" alt="Ax=b" /> trong đó <img src="https://i.upmath.me/svg/A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20n%7D%2C%20x%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D%2C%20b%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D%20" alt="A \in \mathbb{R}^{n \times n}, x \in \mathbb{R}^{n}, b \in \mathbb{R}^{n} " />.  
 
<img src="https://i.upmath.me/svg/Ax%3Db" alt="Ax=b" /> 

<img src="https://i.upmath.me/svg/LUx%3Db" alt="LUx=b" />

<img src="https://i.upmath.me/svg/Ux%3DL%5E%7B-1%7Db" alt="Ux=L^{-1}b" />

<img src="https://i.upmath.me/svg/x%3DU%5E%7B-1%7D(L%5E%7B-1%7Db)" alt="x=U^{-1}(L^{-1}b)" />  

Cài đặt bằng Python, sử dụng hàm LU_Decomposition() đã viết ở trên.
```python
def solve_system_equation(A,b):
  L,U=LU_Decomposition(A)
  x=np.linalg.inv(U).dot((np.linalg.inv(L).dot(b)))
  return x
```
```python
A=np.array([[3,2,-1],[2,-2,4],[-1, 0.5, -1]], dtype=np.float32)
b=np.array([1,-2,0])
x=solve_system_equation(A,b)
print("x=",x)
```
Kết quả chạy  
Hệ có nghiệm
```python
x= [ 1.00000057 -2.00000129 -2.00000103]
```
# 2. Tính định thức 
Với ma trận vuông khả nghịch <img src="https://i.upmath.me/svg/A" alt="A" /> ta có <img src="https://i.upmath.me/svg/det(A)%3Ddet(LU)%3Ddet(L).det(U)" alt="det(A)=det(LU)=det(L).det(U)" />  

Mà <img src="https://i.upmath.me/svg/L%2C%20U" alt="L, U" /> là các ma trận tam giác nên định thức bằng tích các phần tử trên đường chéo chính và đường chéo chính của <img src="https://i.upmath.me/svg/L" alt="L" /> đều là số <img src="https://i.upmath.me/svg/1" alt="1" /> nên <img src="https://i.upmath.me/svg/det(L)%3D1" alt="det(L)=1" />  
Do đó <img src="https://i.upmath.me/svg/det(A)%3Ddet(U)" alt="det(A)=det(U)" />  

Ví dụ với ma trận <img src="https://i.upmath.me/svg/A%3D%5Cbegin%7Bbmatrix%7D%208%20%26%202%20%26%209%20%5C%5C%204%26%209%20%26%204%20%5C%5C%206%20%26%207%20%20%26%209%20%5Cend%7Bbmatrix%7D" alt="A=\begin{bmatrix} 8 &amp; 2 &amp; 9 \\ 4&amp; 9 &amp; 4 \\ 6 &amp; 7  &amp; 9 \end{bmatrix}" />, như đã làm ở trên thì <img src="https://i.upmath.me/svg/A" alt="A" /> có phân tích <img src="https://i.upmath.me/svg/LU" alt="LU" /> là  
<img src="https://i.upmath.me/svg/A%3D%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%200%20%5C%5C%20%5Cdfrac%7B1%7D%7B2%7D%26%201%20%26%200%20%5C%5C%20%5Cdfrac%7B3%7D%7B4%7D%20%26%20%5Cdfrac%7B11%7D%7B16%7D%20%20%26%201%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%208%20%26%202%20%26%209%20%5C%5C%200%20%26%208%20%26%20%5Cdfrac%7B-1%7D%7B2%7D%20%5C%5C%200%20%26%200%20%20%26%20%5Cdfrac%7B83%7D%7B32%7D%20%5Cend%7Bbmatrix%7D" alt="A=\begin{bmatrix} 1 &amp; 0 &amp; 0 \\ \dfrac{1}{2}&amp; 1 &amp; 0 \\ \dfrac{3}{4} &amp; \dfrac{11}{16}  &amp; 1 \end{bmatrix} \begin{bmatrix} 8 &amp; 2 &amp; 9 \\ 0 &amp; 8 &amp; \dfrac{-1}{2} \\ 0 &amp; 0  &amp; \dfrac{83}{32} \end{bmatrix}" />  
Khi đó <img src="https://i.upmath.me/svg/det(A)%3D%5Cbegin%7Bvmatrix%7D%208%20%26%202%20%26%209%20%5C%5C%200%20%26%208%20%26%20%5Cdfrac%7B-1%7D%7B2%7D%20%5C%5C%200%20%26%200%20%20%26%20%5Cdfrac%7B83%7D%7B32%7D%20%5Cend%7Bvmatrix%7D%3D%208.8.%5Cdfrac%7B83%7D%7B32%7D%3D166" alt="det(A)=\begin{vmatrix} 8 &amp; 2 &amp; 9 \\ 0 &amp; 8 &amp; \dfrac{-1}{2} \\ 0 &amp; 0  &amp; \dfrac{83}{32} \end{vmatrix}= 8.8.\dfrac{83}{32}=166" />  


Kiểm tra lại bằng Python  
```python
def determinant_of_matrix(A):
  L,U=LU_Decomposition(A)
  det=np.linalg.det(U)
  print("U=",U)
  return det
```
```python
A=np.array([[8.,2.,9.],[4.,9.,4.],[6.,7.,9.]])
print("Det by library: ",np.linalg.det(A))
print("Det by LU Decomposition: ",determinant_of_matrix(A))
```
Kết quả chạy  
```python
Det by library:  165.99999999999991
U= [[ 8.       2.       9.     ]
 [ 0.       8.      -0.5    ]
 [ 0.       0.       2.59375]]
Det by LU Decomposition:  165.99999999999991
```
Trong trường hợp <img src="https://i.upmath.me/svg/A" alt="A" /> là ma trận xác định dương (positive definite matrix) và <img src="https://i.upmath.me/svg/U%3D%20L%5ET" alt="U= L^T" /> với <img src="https://i.upmath.me/svg/L" alt="L" /> là ma trận tam giác dưới với các thành phần trên đường chéo chính là các số thực dương thì ta có *phân tích Cholesky*, mình sẽ trình bày ở bài sau.


# IV. Tài liệu tham khảo 
1. <https://courses.grainger.illinois.edu/cs357/sp2020/notes/ref-9-linsys.html>
2. <https://vi.wikipedia.org/wiki/Ph%C3%A2n_t%C3%ADch_LU>
3. <https://fit.mta.edu.vn/files/DanhSach/BaigiangDSTT_16.pdf>
4. <http://www.seas.ucla.edu/~vandenbe/133A/133A-notes.pdf>
5. <http://www.math.iit.edu/~fass/477577_Chapter_7.pdf>
6. <https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/>





















