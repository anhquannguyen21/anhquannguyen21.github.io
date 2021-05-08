---
layout: post
title: Gaussian Mixture Model
tags: [Machine Learning]
---
  



## I. **Giới thiệu**
K-Means Clustering là một thuật toán thuộc loại unsupervised learning, một trong những thuật toán đơn giản và phổ biến nhất trong Machine Learning, được dùng trong bài toán phân cụm dữ liệu. Tuy nhiên, nó vẫn có một vài hạn chế nhất định. Một trong số đó là Kmeans thường chỉ phù hợp với cluster có dạng hình tròn. Nếu cluster có dạng hình elip dẹt và khi sử dụng Kmeans để phân cụm thì cho kết quả khá tệ. Lúc này thuật toán Gaussian Mixture sẽ phát huy tác dụng của nó. Nếu Kmeans là thuật toán phân cụm dựa trên khoảng cách thì Gaussian Mixture lại dựa trên phân phối xác suất. Thực chất, Kmeans chỉ là một trường hợp cụ thể của Gaussian Mixture. Thuật toán Kmeans chỉ cập nhật giá trị trung bình. Trong khi đó, Gaussian Mixture cập nhật cả trung bình và phương sai. Tiếp theo ta sẽ cùng tìm hiểu kĩ hơn về thuật toán này.
## II. **Gaussian Mixture Model**
### 1. **Mixture Model**
- Mixture model được dùng để mô tả phân phối xác suất <img src="https://i.upmath.me/svg/p(x)" alt="p(x)" /> bởi các tổ hợp lồi (convex combination) của <img src="https://i.upmath.me/svg/K" alt="K" /> phân phối xác suất  
<img src="https://i.upmath.me/svg/p(x)%3D%20%5Cdisplaystyle%20%5Csum_%7Bk%3D1%7D%5EK%20%5Cpi_k%20p_k(x)" alt="p(x)= \displaystyle \sum_{k=1}^K \pi_k p_k(x)" />  
<img src="https://i.upmath.me/svg/0%5Cle%20%5Cpi_k%20%5Cle%201%2C%20%5Cdisplaystyle%20%5Csum_%7Bk%3D1%7D%5EK%20%5Cpi_k%20%3D1%20" alt="0\le \pi_k \le 1, \displaystyle \sum_{k=1}^K \pi_k =1 " />  
trong đó <img src="https://i.upmath.me/svg/p_k" alt="p_k" /> là các phân phối xác suất cơ bản như Gaussians, Bernoullis, Gammas,... và <img src="https://i.upmath.me/svg/%5Cpi_k" alt="\pi_k" /> là các mixture weights.  
### 2. **Gaussian Mixture Model**  
- Một Gaussian Mixture Model là một kết hợp của <img src="https://i.upmath.me/svg/K" alt="K" /> Gaussian distributions <img src="https://i.upmath.me/svg/%5Cmathcal%7BN%7D(x%7C%5Cmu_k%2C%20%5CSigma_k)" alt="\mathcal{N}(x|\mu_k, \Sigma_k)" /> sao cho  
<img src="https://i.upmath.me/svg/p(x%7C%5Ctheta)%20%3D%20%5Cdisplaystyle%20%5Csum_%7Bk%3D1%7D%5EK%20%5Cpi_k%20%5Cmathcal%7BN%7D(x%7C%5Cmu_k%2C%20%5CSigma_k)%20%20" alt="p(x|\theta) = \displaystyle \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)  " />  
<img src="https://i.upmath.me/svg/0%5Cle%20%5Cpi_k%20%5Cle%201%2C%20%5Cdisplaystyle%20%5Csum_%7Bk%3D1%7D%5EK%20%5Cpi_k%20%3D1%20" alt="0\le \pi_k \le 1, \displaystyle \sum_{k=1}^K \pi_k =1 " />  
trong đó <img src="https://i.upmath.me/svg/%5Ctheta%20%3D%20%5C%7B%5Cmu_k%2C%20%5CSigma_k%2C%20%5Cpi_k%3A%20k%3D1%2C...%2CK%5C%7D" alt="\theta = \{\mu_k, \Sigma_k, \pi_k: k=1,...,K\}" /> là tập hợp tất cả các parameters của model.  
[![gmm.png](https://i.postimg.cc/J48hvvST/gmm.png)](https://postimg.cc/jC8KJgxN)  
Hình 1: Ví dụ Gaussian Mixture Model    
<img src="https://i.upmath.me/svg/p(x%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D)%3D0.5%20%5Cmathcal%7BN%7D%5Cleft(x%20%5Cmid-2%2C%20%5Cfrac%7B1%7D%7B2%7D%5Cright)%2B0.2%20%5Cmathcal%7BN%7D(x%20%5Cmid%201%2C2)%2B0.3%20%5Cmathcal%7BN%7D(x%20%5Cmid%204%2C1)" alt="p(x \mid \boldsymbol{\theta})=0.5 \mathcal{N}\left(x \mid-2, \frac{1}{2}\right)+0.2 \mathcal{N}(x \mid 1,2)+0.3 \mathcal{N}(x \mid 4,1)" />  
  
Giả sử ta có tập dữ liệu <img src="https://i.upmath.me/svg/%0A%5Cmathcal%7BX%7D%3D%5Cleft%5C%7B%5Cboldsymbol%7Bx%7D_%7B1%7D%2C%20%5Cldots%2C%20%5Cboldsymbol%7Bx%7D_%7BN%7D%5Cright%5C%7D%0A" alt="
\mathcal{X}=\left\{\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{N}\right\}
" /> trong đó <img src="https://i.upmath.me/svg/%5Cboldsymbol%7Bx%7D_%7Bn%7D" alt="\boldsymbol{x}_{n}" />, <img src="https://i.upmath.me/svg/n%3D1%2C..N" alt="n=1,..N" /> được xây dựng từ phân phối xác suất <img src="https://i.upmath.me/svg/p(%5Cboldsymbol%7Bx%7D)" alt="p(\boldsymbol{x})" /> chưa biết. Mục tiêu của ta là tìm một xấp xỉ (approximation) tốt của <img src="https://i.upmath.me/svg/p(%5Cboldsymbol%7Bx%7D)" alt="p(\boldsymbol{x})" /> bằng GMM với <img src="https://i.upmath.me/svg/K" alt="K" /> mixture components. Các parameters của GMM là <img src="https://i.upmath.me/svg/K" alt="K" /> vector kỳ vọng <img src="https://i.upmath.me/svg/%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D" alt="\boldsymbol{\mu}_{k}" />, ma trận hiệp phương sai <img src="https://i.upmath.me/svg/%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D" alt="\boldsymbol{\Sigma}_{k}" /> và mixture weights <img src="https://i.upmath.me/svg/%5Cpi_k" alt="\pi_k" />. Ta kí hiệu <img src="https://i.upmath.me/svg/%5Cboldsymbol%7B%5Ctheta%7D%3A%3D%0A%5Cleft%5C%7B%5Cpi_%7Bk%7D%2C%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%3A%20k%3D1%2C%20%5Cldots%2C%20K%5Cright%5C%7D" alt="\boldsymbol{\theta}:=
\left\{\pi_{k}, \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}: k=1, \ldots, K\right\}" />  
Ta sẽ đi tìm <img src="https://i.upmath.me/svg/%5Cboldsymbol%7B%5Ctheta%7D" alt="\boldsymbol{\theta}" /> bằng Maximum Likelihood Estimation (MLE).  
Likelihood của dữ liệu:  
<img src="https://i.upmath.me/svg/%0Ap(%5Cmathcal%7BX%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D)%3D%5Cprod_%7Bn%3D1%7D%5E%7BN%7D%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%2C%20%5Cquad%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%3D%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bk%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%0A" alt="
p(\mathcal{X} \mid \boldsymbol{\theta})=\prod_{n=1}^{N} p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right), \quad p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)=\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)
" />  
với mỗi thành phần <img src="https://i.upmath.me/svg/p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)" alt="p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)" /> là Gaussian mixture density.  
Ta có **log-likelihood**:  
<img src="https://i.upmath.me/svg/%0A%5Clog%20p(%5Cmathcal%7BX%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D)%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Clog%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%3D%5Cunderbrace%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Clog%20%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bk%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%7D_%7B%3D%3A%20%5Cmathcal%7BL%7D%7D%20.%0A" alt="
\log p(\mathcal{X} \mid \boldsymbol{\theta})=\sum_{n=1}^{N} \log p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)=\underbrace{\sum_{n=1}^{N} \log \sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}_{=: \mathcal{L}} .
" />
  
Mục tiêu của ta là tìm <img src="https://i.upmath.me/svg/%5Cboldsymbol%7B%5Ctheta%7D_%7B%5Cmathrm%7BML%7D%7D%5E%7B*%7D" alt="\boldsymbol{\theta}_{\mathrm{ML}}^{*}" /> bằng cách maximize log-likehood <img src="https://i.upmath.me/svg/%5Cmathcal%7BL%7D" alt="\mathcal{L}" />. Một cách tự nhiên nhất là lấy đạo hàm <img src="https://i.upmath.me/svg/%5Cmathrm%7Bd%7D%20%5Cmathcal%7BL%7D%20%2F%20%5Cmathrm%7Bd%7D%20%5Cboldsymbol%7B%5Ctheta%7D" alt="\mathrm{d} \mathcal{L} / \mathrm{d} \boldsymbol{\theta}" />, cho bằng <img src="https://i.upmath.me/svg/0" alt="0" /> và giải tìm <img src="https://i.upmath.me/svg/%5Cboldsymbol%7B%5Ctheta%7D" alt="\boldsymbol{\theta}" />. Tuy nhiên, không như các lời giải maximum likelihood estimation cho Linear Regression, hay phân phối Gaussian, Bernoulli, ta không thể thu được dạng closed-form solution. Ta sẽ dùng thuật toán EM (Expectation Maximization, đã trình bày ở bài trước) cho GMM.  
Remark: Một phân phối Gaussian nhiều chiều (Multivariate Gaussian) được định nghĩa bởi công thức:  
<img src="https://i.upmath.me/svg/%0Ap(x%20%3B%20%5Cmu%2C%20%5CSigma)%3D%5Cfrac%7B1%7D%7B(2%20%5Cpi)%5E%7Bn%20%2F%202%7D%7C%5CSigma%7C%5E%7B1%20%2F%202%7D%7D%20%5Cexp%20%5Cleft(-%5Cfrac%7B1%7D%7B2%7D(x-%5Cmu)%5E%7BT%7D%20%5CSigma%5E%7B-1%7D(x-%5Cmu)%5Cright)%0A" alt="
p(x ; \mu, \Sigma)=\frac{1}{(2 \pi)^{n / 2}|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)
" />  
với <img src="https://i.upmath.me/svg/%0Ax%20%5Csim%20%5Cmathcal%7BN%7D(%5Cmu%2C%20%5CSigma)%0A" alt="
x \sim \mathcal{N}(\mu, \Sigma)
" />  
Khi đó ta có:  
<img src="https://i.upmath.me/svg/%0A%5Clog%20%5Cmathcal%7BN%7D(%5Cboldsymbol%7Bx%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D)%3D-%5Cfrac%7BD%7D%7B2%7D%20%5Clog%20(2%20%5Cpi)-%5Cfrac%7B1%7D%7B2%7D%20%5Clog%20%5Coperatorname%7Bdet%7D(%5Cboldsymbol%7B%5CSigma%7D)-%5Cfrac%7B1%7D%7B2%7D(%5Cboldsymbol%7Bx%7D-%5Cboldsymbol%7B%5Cmu%7D)%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5CSigma%7D%5E%7B-1%7D(%5Cboldsymbol%7Bx%7D-%5Cboldsymbol%7B%5Cmu%7D)%20.%0A" alt="
\log \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})=-\frac{D}{2} \log (2 \pi)-\frac{1}{2} \log \operatorname{det}(\boldsymbol{\Sigma})-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) .
" />
Ta tính đạo hàm của hàm log-likelihood ứng với các tham số của GMM <img src="https://i.upmath.me/svg/%0A%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%2C%20%5Cpi_%7Bk%7D%0A" alt="
\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}
" />  
<img src="https://i.upmath.me/svg/%0A%5Cbegin%7Barray%7D%7Bl%7D%0A%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%7D%3D%5Cmathbf%7B0%7D%5E%7B%5Ctop%7D%20%5CLongleftrightarrow%20%5Cdisplaystyle%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cfrac%7B%5Cpartial%20%5Clog%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%7D%3D%5Cmathbf%7B0%7D%5E%7B%5Ctop%7D%2C%20%5C%5C%0A%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%3D%5Cmathbf%7B0%7D%20%5CLongleftrightarrow%20%5Cdisplaystyle%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cfrac%7B%5Cpartial%20%5Clog%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%3D%5Cmathbf%7B0%7D%20%5C%5C%0A%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20%5Cpi_%7Bk%7D%7D%3D0%20%5CLongleftrightarrow%20%5Cdisplaystyle%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cfrac%7B%5Cpartial%20%5Clog%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cpi_%7Bk%7D%7D%3D0%0A%5Cend%7Barray%7D%0A" alt="
\begin{array}{l}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}_{k}}=\mathbf{0}^{\top} \Longleftrightarrow \displaystyle \sum_{n=1}^{N} \frac{\partial \log p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\mu}_{k}}=\mathbf{0}^{\top}, \\
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\Sigma}_{k}}=\mathbf{0} \Longleftrightarrow \displaystyle \sum_{n=1}^{N} \frac{\partial \log p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}}=\mathbf{0} \\
\frac{\partial \mathcal{L}}{\partial \pi_{k}}=0 \Longleftrightarrow \displaystyle \sum_{n=1}^{N} \frac{\partial \log p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \pi_{k}}=0
\end{array}
" />  
  
Dễ thấy rằng  
<img src="https://i.upmath.me/svg/%0A%5Cfrac%7B%5Cpartial%20%5Clog%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Ctheta%7D%7D%3D%5Cfrac%7B1%7D%7Bp%5Cleft(x_%7Bn%7D%20%5Cmid%20%5Ctheta%5Cright)%7D%20%5Cfrac%7B%5Cpartial%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Ctheta%7D%7D%0A" alt="
\frac{\partial \log p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\theta}}=\frac{1}{p\left(x_{n} \mid \theta\right)} \frac{\partial p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\theta}}
" />  
với <img src="https://i.upmath.me/svg/%0A%5Cboldsymbol%7B%5Ctheta%7D%3D%5Cleft%5C%7B%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%2C%20%5Cpi_%7Bk%7D%2C%20k%3D1%2C%20%5Cldots%2C%20K%5Cright%5C%7D%0A" alt="
\boldsymbol{\theta}=\left\{\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}, k=1, \ldots, K\right\}
" /> là model parameters và  
<img src="https://i.upmath.me/svg/%0A%5Cfrac%7B1%7D%7Bp%5Cleft(x_%7Bn%7D%20%5Cmid%20%5Ctheta%5Cright)%7D%3D%5Cfrac%7B1%7D%7B%5Cdisplaystyle%20%5Csum_%7Bj%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bj%7D%20%5Cmathcal%7BN%7D%5Cleft(x_%7Bn%7D%20%5Cmid%20%5Cmu_%7Bj%7D%2C%20%5CSigma_%7Bj%7D%5Cright)%7D%0A" alt="
\frac{1}{p\left(x_{n} \mid \theta\right)}=\frac{1}{\displaystyle \sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(x_{n} \mid \mu_{j}, \Sigma_{j}\right)}
" />  
Sau đây ta sẽ lần lượt tính 3 phương trình đạo hàm ở trên.  
Trước tiên ta có một khái niệm: **Responsibilities**  
  
**Responsibilities**  
Ta định nghĩa đại lượng  
<img src="https://i.upmath.me/svg/%0Ar_%7Bn%20k%7D%3A%3D%5Cfrac%7B%5Cpi_%7Bk%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%7D%7B%5Cdisplaystyle%20%5Csum_%7Bj%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bj%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bj%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bj%7D%5Cright)%7D%0A" alt="
r_{n k}:=\frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\displaystyle \sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)}
" />  
là *responsibility* của miture component thứ <img src="https://i.upmath.me/svg/k" alt="k" /> của điểm dữ liệu <img src="https://i.upmath.me/svg/n" alt="n" />.  
<img src="https://i.upmath.me/svg/%0Ap%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cpi_%7Bk%7D%2C%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%3D%5Cpi_%7Bk%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%0A" alt="
p\left(\boldsymbol{x}_{n} \mid \pi_{k}, \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)=\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)
" />  
Ta có <img src="https://i.upmath.me/svg/%0A%5Cboldsymbol%7Br%7D_%7Bn%7D%3A%3D%5Cleft%5Br_%7Bn%201%7D%2C%20%5Cldots%2C%20r_%7Bn%20K%7D%5Cright%5D%5E%7B%5Ctop%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BK%7D%0A" alt="
\boldsymbol{r}_{n}:=\left[r_{n 1}, \ldots, r_{n K}\right]^{\top} \in \mathbb{R}^{K}
" /> là probability vector,  <img src="https://i.upmath.me/svg/%0A%5Csum_%7Bk%7D%20r_%7Bn%20k%7D%3D1%0A" alt="
\sum_{k} r_{n k}=1
" /> và <img src="https://i.upmath.me/svg/r_%7Bnk%7D%20%5Cge%200" alt="r_{nk} \ge 0" />. Nếu <img src="https://i.upmath.me/svg/r_k" alt="r_k" /> lớn có nghĩa là tỉ lệ điểm dữ liệu <img src="https://i.upmath.me/svg/n" alt="n" /> được assign vào component thứ <img src="https://i.upmath.me/svg/k" alt="k" /> lớn.  
#### a) Updating the Means  
**Định lý 1** (Update of the GMM Means). Update của GMM Means được cho bởi công thức:  
<img src="https://i.upmath.me/svg/%0A%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5E%7B%5Ctext%20%7Bnew%20%7D%7D%3D%5Cfrac%7B%5Cdisplaystyle%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%20%5Cboldsymbol%7Bx%7D_%7Bn%7D%7D%7B%5Cdisplaystyle%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%7D%0A" alt="
\boldsymbol{\mu}_{k}^{\text {new }}=\frac{\displaystyle \sum_{n=1}^{N} r_{n k} \boldsymbol{x}_{n}}{\displaystyle \sum_{n=1}^{N} r_{n k}}
" />  
trong đó <img src="https://i.upmath.me/svg/r_%7Bnk%7D" alt="r_{nk}" /> được định nghĩa ở trên.  
**Chứng minh:**  
Ta có:  
<img src="https://i.upmath.me/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%7D%20%26%3D%5Csum_%7Bj%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bj%7D%20%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bj%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bj%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%7D%3D%5Cpi_%7Bk%7D%20%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%7D%20%5C%5C%0A%26%3D%5Cpi_%7Bk%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%2C%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\frac{\partial p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\mu}_{k}} &amp;=\sum_{j=1}^{K} \pi_{j} \frac{\partial \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)}{\partial \boldsymbol{\mu}_{k}}=\pi_{k} \frac{\partial \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\partial \boldsymbol{\mu}_{k}} \\
&amp;=\pi_{k}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right),
\end{aligned}
" />  
Từ đó ta có:  
<img src="https://i.upmath.me/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%7D%20%26%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cfrac%7B%5Cpartial%20%5Clog%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%7D%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cfrac%7B1%7D%7Bp%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Ctheta%5Cright)%7D%20%5Cfrac%7B%5Cpartial%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%7D%20%5C%5C%0A%26%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%20%5Cunderbrace%7B%5Cfrac%7B%5Cpi_%7Bk%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%7D%7B%5Csum_%7Bj%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bj%7D%20%5Cmathcal%7BN%7D%5Cleft(x_%7Bn%7D%20%5Cmid%20%5Cmu_%7Bj%7D%2C%20%5CSigma_%7Bj%7D%5Cright)%7D%7D_%7B%3Dr_%7Bn%20k%7D%7D%20%5C%5C%0A%26%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}_{k}} &amp;=\sum_{n=1}^{N} \frac{\partial \log p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\mu}_{k}}=\sum_{n=1}^{N} \frac{1}{p\left(\boldsymbol{x}_{n} \mid \theta\right)} \frac{\partial p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\mu}_{k}} \\
&amp;=\sum_{n=1}^{N}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1} \underbrace{\frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(x_{n} \mid \mu_{j}, \Sigma_{j}\right)}}_{=r_{n k}} \\
&amp;=\sum_{n=1}^{N} r_{n k}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}
\end{aligned}
" />  
Ta giải phương trình  <img src="https://i.upmath.me/svg/%0A%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%5Cleft(%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5E%7B%5Ctext%20%7Bnew%20%7D%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%7D%3D%5Cmathbf%7B0%7D%5E%7B%5Ctop%7D%0A" alt="
\frac{\partial \mathcal{L}\left(\boldsymbol{\mu}_{k}^{\text {new }}\right)}{\partial \boldsymbol{\mu}_{k}}=\mathbf{0}^{\top}
" />  
<img src="https://i.upmath.me/svg/%0A%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%20%5Cboldsymbol%7Bx%7D_%7Bn%7D%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5E%7B%5Ctext%20%7Bnew%20%7D%7D%20%5CLongleftrightarrow%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5E%7B%5Ctext%20%7Bnew%20%7D%7D%3D%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%20%5Cboldsymbol%7Bx%7D_%7Bn%7D%7D%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%7D%3D%5Cfrac%7B1%7D%7B%7BN_%7Bk%7D%7D%7D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%20%5Cboldsymbol%7Bx%7D_%7Bn%7D%0A" alt="
\sum_{n=1}^{N} r_{n k} \boldsymbol{x}_{n}=\sum_{n=1}^{N} r_{n k} \boldsymbol{\mu}_{k}^{\text {new }} \Longleftrightarrow \boldsymbol{\mu}_{k}^{\text {new }}=\frac{\sum_{n=1}^{N} r_{n k} \boldsymbol{x}_{n}}{\sum_{n=1}^{N} r_{n k}}=\frac{1}{{N_{k}}} \sum_{n=1}^{N} r_{n k} \boldsymbol{x}_{n}
" />  
với <img src="https://i.upmath.me/svg/%0AN_%7Bk%7D%3A%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%0A" alt="
N_{k}:=\sum_{n=1}^{N} r_{n k}
" />  
#### b) Updating the Covariances  
**Định lý 2** (Updates of the GMM Covariances). Update của ma trận hiệp phương sai
 <img src="https://i.upmath.me/svg/%0A%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%2C%20k%3D1%2C%20%5Cldots%2C%20K%0A" alt="
\boldsymbol{\Sigma}_{k}, k=1, \ldots, K
" /> của GMM được cho bởi  
<img src="https://i.upmath.me/svg/%0A%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7Bn%20e%20w%7D%3D%5Cfrac%7B1%7D%7BN_%7Bk%7D%7D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%0A" alt="
\boldsymbol{\Sigma}_{k}^{n e w}=\frac{1}{N_{k}} \sum_{n=1}^{N} r_{n k}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top}
" />  
trong đó <img src="https://i.upmath.me/svg/r_%7Bn%20k%7D" alt="r_{n k}" /> và <img src="https://i.upmath.me/svg/N_k" alt="N_k" /> đã được định nghĩa ở trên.   
**Chứng minh**  
Ta có:  
<img src="https://i.upmath.me/svg/%0A%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cfrac%7B%5Cpartial%20%5Clog%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cfrac%7B1%7D%7Bp%5Cleft(x_%7Bn%7D%20%5Cmid%20%5Ctheta%5Cright)%7D%20%5Cfrac%7B%5Cpartial%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%0A" alt="
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\Sigma}_{k}}=\sum_{n=1}^{N} \frac{\partial \log p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}}=\sum_{n=1}^{N} \frac{1}{p\left(x_{n} \mid \theta\right)} \frac{\partial p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}}
" />  
  

<img src="https://i.upmath.me/svg/%0A%5Cbegin%7Barray%7D%7Bl%7D%0A%5Cfrac%7B%5Cpartial%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%20%5C%5C%0A%3D%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%5Cleft(%5Cpi_%7Bk%7D(2%20%5Cpi)%5E%7B-%5Cfrac%7BD%7D%7B2%7D%7D%20%5Coperatorname%7Bdet%7D%5Cleft(%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%20%5Cexp%20%5Cleft(-%5Cfrac%7B1%7D%7B2%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cright)%5Cright)%20%5C%5C%0A%3D%5Cpi_%7Bk%7D(2%20%5Cpi)%5E%7B-%5Cfrac%7BD%7D%7B2%7D%7D%5Cleft%5B%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%20%5Coperatorname%7Bdet%7D%5Cleft(%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%20%5Cexp%20%5Cleft(-%5Cfrac%7B1%7D%7B2%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cright)%5Cright.%20%5C%5C%0A%5Cleft.%5Cquad%2B%5Coperatorname%7Bdet%7D%5Cleft(%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%20%5Cexp%20%5Cleft(-%5Cfrac%7B1%7D%7B2%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cright)%5Cright%5D%0A%5Cend%7Barray%7D%0A" alt="
\begin{array}{l}
\frac{\partial p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}} \\
=\frac{\partial}{\partial \boldsymbol{\Sigma}_{k}}\left(\pi_{k}(2 \pi)^{-\frac{D}{2}} \operatorname{det}\left(\boldsymbol{\Sigma}_{k}\right)^{-\frac{1}{2}} \exp \left(-\frac{1}{2}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\right)\right) \\
=\pi_{k}(2 \pi)^{-\frac{D}{2}}\left[\frac{\partial}{\partial \boldsymbol{\Sigma}_{k}} \operatorname{det}\left(\boldsymbol{\Sigma}_{k}\right)^{-\frac{1}{2}} \exp \left(-\frac{1}{2}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\right)\right. \\
\left.\quad+\operatorname{det}\left(\boldsymbol{\Sigma}_{k}\right)^{-\frac{1}{2}} \frac{\partial}{\partial \boldsymbol{\Sigma}_{k}} \exp \left(-\frac{1}{2}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\right)\right]
\end{array}
" />  
Mặt khác ta có:  
<img src="https://i.upmath.me/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%20%5Coperatorname%7Bdet%7D%5Cleft(%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%20%26%3D-%5Cfrac%7B1%7D%7B2%7D%20%5Coperatorname%7Bdet%7D%5Cleft(%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%2C%20%5C%5C%0A%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%20%26%3D-%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\Sigma}_{k}} \operatorname{det}\left(\boldsymbol{\Sigma}_{k}\right)^{-\frac{1}{2}} &amp;=-\frac{1}{2} \operatorname{det}\left(\boldsymbol{\Sigma}_{k}\right)^{-\frac{1}{2}} \boldsymbol{\Sigma}_{k}^{-1}, \\
\frac{\partial}{\partial \boldsymbol{\Sigma}_{k}}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right) &amp;=-\boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}
\end{aligned}
" />  
Từ đây ta suy ra được  
<img src="https://i.upmath.me/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%3D%26%20%5Cpi_%7Bk%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%20%5C%5C%0A%26%20%5Ccdot%5Cleft%5B-%5Cfrac%7B1%7D%7B2%7D%5Cleft(%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D-%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cright)%5Cright%5D%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\frac{\partial p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}}=&amp; \pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right) \\
&amp; \cdot\left[-\frac{1}{2}\left(\boldsymbol{\Sigma}_{k}^{-1}-\boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\right)\right]
\end{aligned}
" />  
Khi đó  
<img src="https://i.upmath.me/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20%5Cmathcal%7BL%7D%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%20%26%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cfrac%7B%5Cpartial%20%5Clog%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cfrac%7B1%7D%7Bp%5Cleft(x_%7Bn%7D%20%5Cmid%20%5Ctheta%5Cright)%7D%20%5Cfrac%7B%5Cpartial%20p%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Ctheta%7D%5Cright)%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%7D%20%5C%5C%0A%26%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cunderbrace%7B%5Cfrac%7B%5Cpi_%7Bk%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%7D%7B%5Csum_%7Bj%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bj%7D%20%5Cmathcal%7BN%7D%5Cleft(x_%7Bn%7D%20%5Cmid%20%5Cmu_%7Bj%7D%2C%20%5CSigma_%7Bj%7D%5Cright)%7D%7D_%7B%3Dr_%7Bn%20k%7D%7D%20%5C%5C%0A%26%20%5Ccdot%5Cleft%5B-%5Cfrac%7B1%7D%7B2%7D%5Cleft(%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D-%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cright)%5Cright%5D%20%5C%5C%0A%26%3D-%5Cfrac%7B1%7D%7B2%7D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%5Cleft(%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D-%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cright)%20%5C%5C%0A%26%3D-%5Cfrac%7B1%7D%7B2%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%20%5Cunderbrace%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%7D_%7B%3DN_%7Bk%7D%7D%2B%5Cfrac%7B1%7D%7B2%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cleft(%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%5Cright)%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%20.%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\Sigma}_{k}} &amp;=\sum_{n=1}^{N} \frac{\partial \log p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}}=\sum_{n=1}^{N} \frac{1}{p\left(x_{n} \mid \theta\right)} \frac{\partial p\left(\boldsymbol{x}_{n} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}} \\
&amp;=\sum_{n=1}^{N} \underbrace{\frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(x_{n} \mid \mu_{j}, \Sigma_{j}\right)}}_{=r_{n k}} \\
&amp; \cdot\left[-\frac{1}{2}\left(\boldsymbol{\Sigma}_{k}^{-1}-\boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\right)\right] \\
&amp;=-\frac{1}{2} \sum_{n=1}^{N} r_{n k}\left(\boldsymbol{\Sigma}_{k}^{-1}-\boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\right) \\
&amp;=-\frac{1}{2} \boldsymbol{\Sigma}_{k}^{-1} \underbrace{\sum_{n=1}^{N} r_{n k}}_{=N_{k}}+\frac{1}{2} \boldsymbol{\Sigma}_{k}^{-1}\left(\sum_{n=1}^{N} r_{n k}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top}\right) \boldsymbol{\Sigma}_{k}^{-1} .
\end{aligned}
" />  
Giải phương trình đạo hàm bằng <img src="https://i.upmath.me/svg/0" alt="0" />  
<img src="https://i.upmath.me/svg/%0A%5Cbegin%7Baligned%7D%0AN_%7Bk%7D%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%3D%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%5Cleft(%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%5Cright)%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%20%5C%5C%0A%5CLongleftrightarrow%20N_%7Bk%7D%20%5Cboldsymbol%7BI%7D%3D%5Cleft(%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%5Cright)%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B-1%7D%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
N_{k} \boldsymbol{\Sigma}_{k}^{-1}=\boldsymbol{\Sigma}_{k}^{-1}\left(\sum_{n=1}^{N} r_{n k}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top}\right) \boldsymbol{\Sigma}_{k}^{-1} \\
\Longleftrightarrow N_{k} \boldsymbol{I}=\left(\sum_{n=1}^{N} r_{n k}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top}\right) \boldsymbol{\Sigma}_{k}^{-1}
\end{aligned}
" />  
Cuối cùng ta thu được  

<img src="https://i.upmath.me/svg/%0A%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5E%7B%5Cmathrm%7Bnew%7D%7D%3D%5Cfrac%7B1%7D%7BN_%7Bk%7D%7D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%0A" alt="
\boldsymbol{\Sigma}_{k}^{\mathrm{new}}=\frac{1}{N_{k}} \sum_{n=1}^{N} r_{n k}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top}
" />  
  
#### c) Updating the Mixture Weights   
**Định lý 3** (Update of the GMM Mixture Weights). Các mixture weights của GMM model được cập nhật bởi  
<img src="https://i.upmath.me/svg/%0A%5Cpi_%7Bk%7D%5E%7B%5Ctext%20%7Bnew%20%7D%7D%3D%5Cfrac%7BN_%7Bk%7D%7D%7BN%7D%2C%20%5Cquad%20k%3D1%2C%20%5Cldots%2C%20K%0A" alt="
\pi_{k}^{\text {new }}=\frac{N_{k}}{N}, \quad k=1, \ldots, K
" />  
với <img src="https://i.upmath.me/svg/N" alt="N" /> là số điểm dữ liệu và <img src="https://i.upmath.me/svg/N_k" alt="N_k" /> đã được định nghĩa ở trên.  
**Chứng minh**  
Đây là một bài toán tối ưu có ràng buộc <img src="https://i.upmath.me/svg/%0A%5Csum_%7Bk%7D%20%5Cpi_%7Bk%7D%3D1%0A" alt="
\sum_{k} \pi_{k}=1
" />  
Ta sẽ sử dụng phương pháp nhân tử Lagrange.  
The Lagrangian is  
<img src="https://i.upmath.me/svg/%0A%5Cmathfrak%7BL%7D%3D%5Cmathcal%7BL%7D%2B%5Clambda%5Cleft(%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bk%7D-1%5Cright)%0A" alt="
\mathfrak{L}=\mathcal{L}+\lambda\left(\sum_{k=1}^{K} \pi_{k}-1\right)
" />  
<img src="https://i.upmath.me/svg/%0A%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Clog%20%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bk%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%2B%5Clambda%5Cleft(%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bk%7D-1%5Cright)%0A" alt="
=\sum_{n=1}^{N} \log \sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)+\lambda\left(\sum_{k=1}^{K} \pi_{k}-1\right)
" />  
Khi đó ta có  
<img src="https://i.upmath.me/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20%5Cmathfrak%7BL%7D%7D%7B%5Cpartial%20%5Cpi_%7Bk%7D%7D%20%26%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cfrac%7B%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%7D%7B%5Csum_%7Bj%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bj%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bj%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bj%7D%5Cright)%7D%2B%5Clambda%20%5C%5C%0A%26%3D%5Cfrac%7B1%7D%7B%5Cpi_%7Bk%7D%7D%20%5Cunderbrace%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cfrac%7B%5Cpi_%7Bk%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%7D%7B%5Csum_%7Bj%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bj%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bj%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bj%7D%5Cright)%7D%7D_%7B%3DN_%7Bk%7D%7D%2B%5Clambda%3D%5Cfrac%7BN_%7Bk%7D%7D%7B%5Cpi_%7Bk%7D%7D%2B%5Clambda%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\frac{\partial \mathfrak{L}}{\partial \pi_{k}} &amp;=\sum_{n=1}^{N} \frac{\mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)}+\lambda \\
&amp;=\frac{1}{\pi_{k}} \underbrace{\sum_{n=1}^{N} \frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)}}_{=N_{k}}+\lambda=\frac{N_{k}}{\pi_{k}}+\lambda
\end{aligned}
" />  
và  
<img src="https://i.upmath.me/svg/%0A%5Cfrac%7B%5Cpartial%20%5Cmathfrak%7BL%7D%7D%7B%5Cpartial%20%5Clambda%7D%3D%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bk%7D-1%0A" alt="
\frac{\partial \mathfrak{L}}{\partial \lambda}=\sum_{k=1}^{K} \pi_{k}-1
" />  
Giải phương trình đạo hàm bằng 0  
<img src="https://i.upmath.me/svg/%0A%5Cbegin%7Barray%7D%7Bl%7D%0A%5Cpi_%7Bk%7D%3D-%5Cfrac%7BN_%7Bk%7D%7D%7B%5Clambda%7D%20%5C%5C%0A1%3D%5Cdisplaystyle%20%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bk%7D%0A%5Cend%7Barray%7D%0A" alt="
\begin{array}{l}
\pi_{k}=-\frac{N_{k}}{\lambda} \\
1=\displaystyle \sum_{k=1}^{K} \pi_{k}
\end{array}
" />  
<img src="https://i.upmath.me/svg/%0A%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cpi_%7Bk%7D%3D1%20%5CLongleftrightarrow-%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cfrac%7BN_%7Bk%7D%7D%7B%5Clambda%7D%3D1%20%5CLongleftrightarrow-%5Cfrac%7BN%7D%7B%5Clambda%7D%3D1%20%5CLongleftrightarrow%20%5Clambda%3D-N%0A" alt="
\sum_{k=1}^{K} \pi_{k}=1 \Longleftrightarrow-\sum_{k=1}^{K} \frac{N_{k}}{\lambda}=1 \Longleftrightarrow-\frac{N}{\lambda}=1 \Longleftrightarrow \lambda=-N
" />  
Vậy <img src="https://i.upmath.me/svg/%0A%5Cpi_%7Bk%7D%5E%7B%5Ctext%20%7Bnew%20%7D%7D%3D%5Cfrac%7BN_%7Bk%7D%7D%7BN%7D%0A" alt="
\pi_{k}^{\text {new }}=\frac{N_{k}}{N}
" />  
Ta sẽ sử dụng thuật toán EM để tìm <img src="https://i.upmath.me/svg/%0A%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%2C%20%5Cpi_%7Bk%7D%0A" alt="
\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}
" />  
Đầu tiên ta sẽ init value cho <img src="https://i.upmath.me/svg/%0A%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%2C%20%5Cpi_%7Bk%7D%0A" alt="
\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}
" /> và lặp thuật toán cho đến khi hội tụ.  
- **E-step**: Evaluate the responsibilities <img src="https://i.upmath.me/svg/r_%7Bnk%7D" alt="r_{nk}" /> (posterior probability of data point n belonging to mixture component <img src="https://i.upmath.me/svg/k" alt="k" />).  
- **M-step**: Use the updated responsibilities to reestimate the parameters <img src="https://i.upmath.me/svg/%0A%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%2C%20%5Cpi_%7Bk%7D%0A" alt="
\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}
" />  
Tóm tắt thuật toán như sau:  
1. Initialize <img src="https://i.upmath.me/svg/%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%2C%20%5Cpi_%7Bk%7D" alt="\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}" />.
2. E-step: Evaluate responsibilities <img src="https://i.upmath.me/svg/r_%7Bn%20k%7D" alt="r_{n k}" /> for every data point <img src="https://i.upmath.me/svg/%5Cboldsymbol%7Bx%7D_%7Bn%7D" alt="\boldsymbol{x}_{n}" /> using current parametters <img src="https://i.upmath.me/svg/%5Cpi_%7Bk%7D%2C%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D" alt="\pi_{k}, \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}" />:  
<img src="https://i.upmath.me/svg/%0Ar_%7Bn%20k%7D%3D%5Cfrac%7B%5Cpi_%7Bk%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%5Cright)%7D%7B%5Csum_%7Bj%7D%20%5Cpi_%7Bj%7D%20%5Cmathcal%7BN%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5Cmid%20%5Cboldsymbol%7B%5Cmu%7D_%7Bj%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bj%7D%5Cright)%7D%0A" alt="
r_{n k}=\frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{j} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)}
" />
3. M-step: Reestimate parameters <img src="https://i.upmath.me/svg/%5Cpi_%7Bk%7D%2C%20%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%2C%20%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D" alt="\pi_{k}, \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}" /> using the current responsibilities <img src="https://i.upmath.me/svg/r_%7Bn%20k%7D" alt="r_{n k}" /> (from E-step):  
<img src="https://i.upmath.me/svg/%0A%5Cbegin%7Baligned%7D%0A%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%20%26%3D%5Cfrac%7B1%7D%7BN_%7Bk%7D%7D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%20%5Cboldsymbol%7Bx%7D_%7Bn%7D%20%5C%5C%0A%5Cboldsymbol%7B%5CSigma%7D_%7Bk%7D%20%26%3D%5Cfrac%7B1%7D%7BN_%7Bk%7D%7D%20%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20r_%7Bn%20k%7D%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5Cleft(%5Cboldsymbol%7Bx%7D_%7Bn%7D-%5Cboldsymbol%7B%5Cmu%7D_%7Bk%7D%5Cright)%5E%7B%5Ctop%7D%2C%20%5C%5C%0A%5Cpi_%7Bk%7D%20%26%3D%5Cfrac%7BN_%7Bk%7D%7D%7BN%7D%0A%5Cend%7Baligned%7D%0A" alt="
\begin{aligned}
\boldsymbol{\mu}_{k} &amp;=\frac{1}{N_{k}} \sum_{n=1}^{N} r_{n k} \boldsymbol{x}_{n} \\
\boldsymbol{\Sigma}_{k} &amp;=\frac{1}{N_{k}} \sum_{n=1}^{N} r_{n k}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top}, \\
\pi_{k} &amp;=\frac{N_{k}}{N}
\end{aligned}
" />
 [![gmm1.png](https://i.postimg.cc/GhpkCVsj/gmm1.png)](https://postimg.cc/HjfJ0SzJ)  
[![gmm2.png](https://i.postimg.cc/RFXNpCGx/gmm2.png)](https://postimg.cc/jwnxLbGM)  
[![gmm3.png](https://i.postimg.cc/442Yzdmz/gmm3.png)](https://postimg.cc/23vjYCd6)  
Hình 2: Minh họa thuật toán EM cho Gaussian Mixture Model  

### 3. **Cài đặt với Python**  
#### a) Import các thư viện cần thiết  
```python
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal
%matplotlib inline
```  
  
#### b) Generate data
```python
def generate_data(num_data, means, covariances, weights):
    """ Creates a list of data points """
    num_clusters = len(weights)
    data = []
    for i in range(num_data):
        #  Use np.random.choice and weights to pick a cluster id greater than or equal to 0 and less than num_clusters.
        k = np.random.choice(len(weights), 1, p=weights)[0]

        # Use np.random.multivariate_normal to create data from this cluster
        x = np.random.multivariate_normal(means[k], covariances[k])

        data.append(x)
    return data
```  
  
```python
# Model parameters
init_means = [
    [5, 0], # mean of cluster 1
    [1, 1], # mean of cluster 2
    [0, 5]  # mean of cluster 3
]
init_covariances = [
    [[.5, 0.], [0, .5]], # covariance of cluster 1
    [[.92, .38], [.38, .91]], # covariance of cluster 2
    [[.5, 0.], [0, .5]]  # covariance of cluster 3
]
init_weights = [1/4., 1/2., 1/4.]  # weights of each cluster

# Generate data
np.random.seed(4)
data = generate_data(400, init_means, init_covariances, init_weights)
```  
  

```python
#Visualize dữ liệu
plt.figure()
d = np.vstack(data)
plt.plot(d[:,0], d[:,1],'ko')
plt.rcParams.update({'font.size':16})
plt.tight_layout()
```  
  
  
[![gmm.png](https://i.postimg.cc/0yM4NCNK/gmm.png)](https://postimg.cc/HrmzZXXT)  
  
#### c) Cài đặt GMM
```python
#Tính log-likelihood của dữ liệu
def log_sum_exp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

def loglikelihood(data, weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
    num_clusters = len(means)
    num_dim = len(data[0])
    
    ll = 0
    for d in data:
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            
            # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            
            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1/2. * (num_dim * np.log(2*np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
            
        # Increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)
        
    return ll
```
  
```python
#Thuật toán EM
def EM(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4):
    
    # Make copies of initial parameters, which we will update during each iteration
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]
    
    # Infer dimensions of dataset and the number of clusters
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)
    
    # Initialize some useful variables
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]
    
    for i in range(maxiter):
        if i % 5 == 0:
            print("Iteration %s" % i)
        
        # E-step: compute responsibilities
        # Update resp matrix so that resp[j, k] is the responsibility of cluster k for data point j.
        # Hint: To compute likelihood of seeing data point j given cluster k, use multivariate_normal.pdf.
        for j in range(num_data):
            for k in range(num_clusters):
                resp[j, k] =  weights[k] * multivariate_normal.pdf(data[j], mean=means[k], cov=covariances[k])
        row_sums = resp.sum(axis=1)[:, np.newaxis]
        resp = resp / row_sums # normalize over all possible cluster assignments

        # M-step
        # Compute the total responsibility assigned to each cluster, which will be useful when 
        # implementing M-steps below. In the lectures this is called N^{soft}
        counts = np.sum(resp, axis=0)
        
        for k in range(num_clusters):
            
            # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
            weights[k] = counts[k]
            
            # Update means for cluster k using the M-step update rule for the mean variables.
            # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
            
            weighted_sum = 0
            for j in range(num_data):
                weighted_sum += data[j] * resp[j,k]
            means[k] = weighted_sum / weights[k]
            
            # Update covariances for cluster k using the M-step update rule for covariance variables.
            # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
            weighted_sum = np.zeros((num_dim, num_dim))
            for j in range(num_data):
                #(Hint: Use np.outer on the data[j] and this cluster's mean)
                weighted_sum += np.outer(data[j] - means[k],data[j] - means[k]) * resp[j,k]

            covariances[k] = weighted_sum / weights[k]

        
        
        # Compute the loglikelihood at this iteration
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)
        
        # Check for convergence in log-likelihood and store
        if (ll_latest - ll) < thresh and ll_latest > -np.inf:
            break
        ll = ll_latest
    
    if i % 5 != 0:
        print("Iteration %s" % i)
    weights = weights / sum(weights)
    out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}

    return out
```  
  
```python
np.random.seed(4)

# Initialization of parameters
chosen = np.random.choice(len(data), 3, replace=False)
initial_means = [data[x] for x in chosen]
initial_covs = [np.cov(data, rowvar=0)] * 3
initial_weights = [1/3.] * 3
# Run EM 
results = EM(data, initial_means, initial_covs, initial_weights)
```  
    
```python
results['weights']
```  
  
[![gmm.png](https://i.postimg.cc/nVsZmqMZ/gmm.png)](https://postimg.cc/F1Q25djn)  
  

```python
results['means']
```  
  
[![gmm.png](https://i.postimg.cc/HsQq0zmG/gmm.png)](https://postimg.cc/mtgpBQhV)  
  
```python
results['covs'][0]  
```    
  
[![gmm.png](https://i.postimg.cc/g2gB4BB7/gmm.png)](https://postimg.cc/fV0KR5s7)  
  

```python
results['covs'][1]  
```     
    
[![gmm.png](https://i.postimg.cc/RhxPNrNW/gmm.png)](https://postimg.cc/xJgG44GQ)  
  


```python
results['covs'][2]  
```    
  
[![gmm.png](https://i.postimg.cc/0y40zg49/gmm.png)](https://postimg.cc/5XwCrr1r)  
  

```python
def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                 mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom  
```  
  
```python
import matplotlib.mlab as mlab
def plot_contours(data, means, covs, title):
    plt.figure()
    plt.plot([x[0] for x in data], [y[1] for y in data],'ko') # data

    delta = 0.025
    k = len(means)
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1]/(sigmax*sigmay)
        Z = bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
        plt.contour(X, Y, Z, colors = col[i])
        plt.title(title)
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
```  
  
```python
# Parameters after initialization
plot_contours(data, initial_means, initial_covs, 'Initial clusters')
```  
  
[![gmm.png](https://i.postimg.cc/Bb5KpkNP/gmm.png)](https://postimg.cc/8f5cPy2T)  
  
```python
# Parameters after running EM to convergence
results = EM(data, initial_means, initial_covs, initial_weights)
plot_contours(data, results['means'], results['covs'], 'Final clusters')
```  
  
[![gmm.png](https://i.postimg.cc/3N9wdTn1/gmm.png)](https://postimg.cc/T5Lx4Z85)    
  
## III. Tài liệu tham khảo 
1. Book Mathematics for Machine Learning, by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong.  
2. https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/gaussian.pdf  









 
  







