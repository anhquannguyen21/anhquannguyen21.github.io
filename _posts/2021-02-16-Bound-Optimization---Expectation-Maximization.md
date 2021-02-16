---
layout: post
title: Bound Optimization - Expectation Maximization
tags: [Machine Learning]
---



- Trong bài viết này chúng ta sẽ đề cập đến một lớp các thuật toán được gọi là **Bound Optimization** hoặc **MM Optimization**. Trong ngữ cảnh minimization, MM được hiểu là **Majorize-Minimize**, trong ngữ cảnh maximization, MM được hiểu là **Majorize-Maximize**. Chúng ta sẽ nói về một trường hợp đặc biệt của MM là **Expectation Maximization** hoặc **EM**.  
  ### **I. Định nghĩa**
- Giả sử mục tiệu là maximize hàm <img src="https://i.upmath.me/svg/L(%5Ctheta)" alt="L(\theta)" />, chẳng hạn như hàm log likelihood với tham số <img src="https://i.upmath.me/svg/%5Ctheta" alt="\theta" />. Cách tiếp cận cơ bản trong MM là đi xây dựng **surrogate function** <img src="https://i.upmath.me/svg/Q(%5Ctheta%2C%20%5Ctheta%5Et)" alt="Q(\theta, \theta^t)" /> là một chặn dưới của <img src="https://i.upmath.me/svg/L(%5Ctheta)" alt="L(\theta)" />, <img src="https://i.upmath.me/svg/Q(%5Ctheta%2C%20%5Ctheta%5Et)%20%5Cle%20L(%5Ctheta)" alt="Q(\theta, \theta^t) \le L(\theta)" /> và <img src="https://i.upmath.me/svg/Q(%5Ctheta%5Et%2C%20%5Ctheta%5Et)%20%3D%20L(%5Ctheta%5Et)" alt="Q(\theta^t, \theta^t) = L(\theta^t)" />. Nếu các điều kiện này thỏa mãn, ta gọi <img src="https://i.upmath.me/svg/Q" alt="Q" /> **minorizes** <img src="https://i.upmath.me/svg/L" alt="L" />.  
Sau đó thực hiện cập nhật sau mỗi bước:  
<img src="https://i.upmath.me/svg/%5Ctheta%5E%7Bt%2B1%7D%3D%20%5Coperatorname*%7Bargmax%7D_%5Ctheta%20Q(%5Ctheta%2C%20%5Ctheta%5Et)%20" alt="\theta^{t+1}= \operatorname*{argmax}_\theta Q(\theta, \theta^t) " />  
Điều này đảm bảo sự đơn điệu tăng trong hàm mục tiêu ban đầu.   
<img src="https://i.upmath.me/svg/l(%5Ctheta%5E%7Bt%2B1%7D)%20%5Cge%20Q(%5Ctheta%5E%7Bt%2B1%7D%2C%20%5Ctheta%5Et)%20%5Cge%20Q(%5Ctheta%5Et%2C%20%5Ctheta%5Et)%20%3D%20l(%5Ctheta%5Et)" alt="l(\theta^{t+1}) \ge Q(\theta^{t+1}, \theta^t) \ge Q(\theta^t, \theta^t) = l(\theta^t)" />  
[![ep1.png](https://i.postimg.cc/pTShKpPj/ep1.png)](https://postimg.cc/r0Swky0y)  
                       Minh họa Bound Optimization: Đường nét đứt màu đỏ là hàm mục tiêu gốc, đường nét liền màu xanh là lower bound tại vị trí <img src="https://i.upmath.me/svg/%5Ctheta%5Et" alt="\theta^t" />, nó tiếp xúc với hàm objective gốc tại vị trí <img src="https://i.upmath.me/svg/%5Ctheta%5Et" alt="\theta^t" />, và maximum của đường này là <img src="https://i.upmath.me/svg/%5Ctheta%5E%7Bt%2B1%7D" alt="\theta^{t+1}" />. Đường nét đứt màu xanh tiếp xúc với hàm mục tiêu gốc tại <img src="https://i.upmath.me/svg/%5Ctheta%5E%7Bt%2B1%7D" alt="\theta^{t+1}" />, maximum mới là <img src="https://i.upmath.me/svg/%5Ctheta%5E%7Bt%2B2%7D" alt="\theta^{t+2}" />.  
   ### **II. Thuật toán Expectation Maximization (EM)**  

- EM là thuật toán để tính MLE (Maximum Likelihood Estimation) hoặc MAP parameter estimate cho probability models mà có dữ liệu không đầy đủ (missing data) hoặc có biến ẩn (hidden variables).    
*"A general technique for finding maximum likelihood estimators in latent variable models is the expectation-maximization (EM) algorithm."*  
***Page 424, Pattern Recognition and Machine Learning, 2006.***  
*"… if we have missing data and/or latent variables, then computing the [maximum likelihood] estimate becomes hard."*  
***Page 349, Machine Learning: A Probabilistic Perspective, 2012***  
- EM gồm 2 bước chính:  
  1. **E step (Expectation step)**: Estimate the missing variables in the dataset.
  2. **M step (Maximization step)**: Maximize the parameters of the model in the presence of the data.  
- Một số ứng dụng của EM như fit mixture models (such as Gaussian mixture model), fit a multivariate Gaussian (khi missing data), fit robust linear regression models,...  
     #### 1. Lower bound  
- Mục tiêu của thuật toán EM là maximize log likelihood của dữ liệu quan sát được (observed data):  
[![ep2.png](https://i.postimg.cc/HLvXtpmQ/ep2.png)](https://postimg.cc/xkHXfVcC)  
  
    trong đó <img src="https://i.upmath.me/svg/y_n" alt="y_n" /> là dữ liệu quan sát được (visible variables) còn <img src="https://i.upmath.me/svg/z_n" alt="z_n" /> là dũ liệu ẩn (visible variables).  
   Xét một tập hợp các phân phối tùy ý (arbitrary distributions) <img src="https://i.upmath.me/svg/q_n(z_n)" alt="q_n(z_n)" /> đối với mỗi biến ẩn <img src="https://i.upmath.me/svg/z_n" alt="z_n" />, hàm log likelihood có thể viết lại như sau:  
[![ep3.png](https://i.postimg.cc/L51WpCDb/ep3.png)](https://postimg.cc/LgmDkVrt)  
  
     **Nhắc lại bất đẳng thức Jensen (Jensen’s inequality)**: Với hàm lồi (convex function) <img src="https://i.upmath.me/svg/f" alt="f" />, ta có:  
[![ep4.png](https://i.postimg.cc/ncsFspRy/ep4.png)](https://postimg.cc/N5cqSqwD)  
  
  với <img src="https://i.upmath.me/svg/%5Clambda_i%20%5Cge%200" alt="\lambda_i \ge 0" /> và <img src="https://i.upmath.me/svg/%5Cdisplaystyle%20%5Csum_%7Bi%3D1%7D%5En%20%5Clambda_i%3D1" alt="\displaystyle \sum_{i=1}^n \lambda_i=1" />.  
  Đối với hàm lõm (concave function) thì ta đổi <img src="https://i.upmath.me/svg/%5Cle%20" alt="\le " /> thành <img src="https://i.upmath.me/svg/%5Cge" alt="\ge" />. Ví dụ với hàm <img src="https://i.upmath.me/svg/f(z)%3Dlog(z)" alt="f(z)=log(z)" /> là hàm lõm ta có  <img src="https://i.upmath.me/svg/%5Cdisplaystyle%20%5Clog(E_z(g(z)))%20%5Cge%20E_z%20%5Clog(g(z))" alt="\displaystyle \log(E_z(g(z))) \ge E_z \log(g(z))" />.  
  Sử dụng BĐT Jensen ta có:  
[![ep1.png](https://i.postimg.cc/pVmQdnvR/ep1.png)](https://postimg.cc/G91T7t95)
  
   trong đó <img src="https://i.upmath.me/svg/H(q)" alt="H(q)" /> là phân phối xác suất của <img src="https://i.upmath.me/svg/q" alt="q" />:  
  <img src="https://i.upmath.me/svg/%5Cdisplaystyle%20H(q)%3D-%5Csum_z%20q(z)%5Clog%20q(z)" alt="\displaystyle H(q)=-\sum_z q(z)\log q(z)" />  
  Biểu thức <img src="https://i.upmath.me/svg/Q" alt="Q" /> ở dòng cuối cùng được gọi là **evidence lower bound** hoặc **ELBO**. (ELBO sẽ được đề cập tiếp trong bài viết về **Variational Autoencoder**)  
 Thuật toán EM sẽ luân phiên việc tối đa lower bound <img src="https://i.upmath.me/svg/Q" alt="Q" /> wrt phân phối <img src="https://i.upmath.me/svg/q_n" alt="q_n" /> và tham số <img src="https://i.upmath.me/svg/%5Ctheta" alt="\theta" />
   #### 2. E step
- Ta có:  
[![ep1.png](https://i.postimg.cc/7Zt0DxRf/ep1.png)](https://postimg.cc/sMSv4Rsr)  
  
  trong đó  
[![ep1.png](https://i.postimg.cc/fTw99tnV/ep1.png)](https://postimg.cc/QVyCLMMD)  
là **Kullback-Leibler divergence** giữa 2 phân phối xác suất <img src="https://i.upmath.me/svg/p" alt="p" /> và <img src="https://i.upmath.me/svg/q" alt="q" />. Ta có <img src="https://i.upmath.me/svg/KL(p%20%5Cparallel%20q)%20%5Cge%200" alt="KL(p \parallel q) \ge 0" /> và <img src="https://i.upmath.me/svg/KL(p%20%5Cparallel%20q)%20%3D%200%20%5Ciff%20p%3Dq" alt="KL(p \parallel q) = 0 \iff p=q" />.  
DO đó ta có thể maximize lower bound <img src="https://i.upmath.me/svg/Q(%5Ctheta%2C%20q_n)" alt="Q(\theta, q_n)" /> wrt <img src="https://i.upmath.me/svg/q_n" alt="q_n" /> bằng cách cho <img src="https://i.upmath.me/svg/q_n%5E*%3D%20p(z_n%7Cy_n%2C%20%5Ctheta)" alt="q_n^*= p(z_n|y_n, \theta)" />. Khi đó ta có:  
[![ep1.png](https://i.postimg.cc/jqJb4ZT9/ep1.png)](https://postimg.cc/nMZgHKCY)  
  
  #### 3. M step 
- Trong M step, ta cần maximize <img src="https://i.upmath.me/svg/Q(%5Ctheta%2C%5C%7Bq_n%5C%7D)" alt="Q(\theta,\{q_n\})" /> wrt <img src="https://i.upmath.me/svg/%5Ctheta" alt="\theta" />, với <img src="https://i.upmath.me/svg/q_n%5Et" alt="q_n^t" /> được tính ở E step ở iteration t. Do <img src="https://i.upmath.me/svg/H(q_n)" alt="H(q_n)" /> là hằng số đối với <img src="https://i.upmath.me/svg/%5Ctheta" alt="\theta" /> ta có thể bỏ trong bước M step.  Ta thu được:  
[![ep1.png](https://i.postimg.cc/qqtMsTLZ/ep1.png)](https://postimg.cc/1VQQSbbp)  
Biểu thức này được gọi là **expected complete data log likelihood**. Ta sẽ maximize hàm này để thu được  
[![ep1.png](https://i.postimg.cc/ZqRRbpNS/ep1.png)](https://postimg.cc/K16FQKQq)  
  
  Tóm lại, thuật toán EM được viết dạng mã giả như sau (<img src="https://i.upmath.me/svg/x" alt="x" />: mẫu dữ liệu, thông tin về <img src="https://i.upmath.me/svg/y" alt="y" /> bị ẩn).  
[![ep1.png](https://i.postimg.cc/ydsL36Vs/ep1.png)](https://postimg.cc/3yfXPT4c)

  
  Về ứng dụng của EM mình sẽ trình bày ở các bài tiếp theo.  
## III. Tài liệu tham khảo  
1. Probabilistic Machine Learning: An Introduction, by Kevin Patrick Murphy.
MIT Press, 2021.  
https://probml.github.io/pml-book/book1.html  
2. http://uet.vnu.edu.vn/~tqlong/2016hmtk/em.pdf
3. https://machinelearningmastery.com/expectation-maximization-em-algorithm/



  
  


  