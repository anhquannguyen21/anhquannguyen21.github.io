---
layout: post
title: Về tính hội tụ của thuật toán Gradient Descent (Phần 2)
tags: [Convex Optimization]
---




[![B4.png](https://i.postimg.cc/B6N7nH5v/B4.png)](https://postimg.cc/TLLJky48)
  
- Ở bài cuối của chuỗi bài phân tích tính hội tụ của thuật toán Gradient Descent mình sẽ phân tích đối với hàm smooth và strongly convex function. Đây là 2 tính chất quan trọng kết hợp trong 1 hàm làm cho thuật toán gradient descent **"faster"**.
- **Định nghĩa:** Cho hàm <img src="https://i.upmath.me/svg/f%3A%20dom(f)%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: dom(f) \rightarrow \mathbb{R}" /> là hàm lồi và khả vi, <img src="https://i.upmath.me/svg/X%20%5Csubseteq%20dom(f)" alt="X \subseteq dom(f)" /> lồi và <img src="https://i.upmath.me/svg/%20%5Cmu%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B%2B%7D" alt=" \mu \in \mathbb{R}^{+}" />. Hàm <img src="https://i.upmath.me/svg/f" alt="f" /> được gọi là **strongly convex** (với parameter <img src="https://i.upmath.me/svg/%5Cmu" alt="\mu" />) trên <img src="https://i.upmath.me/svg/X" alt="X" /> nếu  
<img src="https://i.upmath.me/svg/f(y)%20%5Cge%20f(x)%20%2B%20%5Cnabla%20f(x)%5ET(y-x)%2B%20%5Cdfrac%7B%5Cmu%7D%7B2%7D%5C%7Cx-y%5C%7C%5E2" alt="f(y) \ge f(x) + \nabla f(x)^T(y-x)+ \dfrac{\mu}{2}\|x-y\|^2" />, <img src="https://i.upmath.me/svg/%5Cforall%20x%2C%20y%20%5Cin%20X" alt="\forall x, y \in X" />.  
Nếu <img src="https://i.upmath.me/svg/X%3Ddom(f)" alt="X=dom(f)" /> thì <img src="https://i.upmath.me/svg/f" alt="f" /> đơn giản gọi là **strongly convex**.  
Hàm smooth convex đã được định nghĩa ở bài trước.  
Dễ thấy rằng mọi hàm lồi đều thỏa mãn tính chất trên với <img src="https://i.upmath.me/svg/%5Cmu%3D0" alt="\mu=0" />.  
[![strongly.png](https://i.postimg.cc/Kz9PXcFc/strongly.png)](https://postimg.cc/5YCYLV6D)  
  
Ta có một số tính chất quan trọng sau của strongly convex function.  
- **Bổ đề 1**: Nếu hàm <img src="https://i.upmath.me/svg/f%3A%20%5Cmathbb%7BR%7D%5Ed%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: \mathbb{R}^d \rightarrow \mathbb{R}" /> strongly convex với tham số <img src="https://i.upmath.me/svg/%5Cmu%20%3E%200" alt="\mu &gt; 0" /> thì <img src="https://i.upmath.me/svg/f" alt="f" /> strictly convex và <img src="https://i.upmath.me/svg/f" alt="f" /> có điểm global minimum duy nhất.  
- **Bổ đề 2**: Cho <img src="https://i.upmath.me/svg/f%3A%20%5Cmathbb%7BR%7D%5Ed%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: \mathbb{R}^d \rightarrow \mathbb{R}" /> strongly convex và smooth convex với tham số <img src="https://i.upmath.me/svg/%5Cmu%20%3E0%20" alt="\mu &gt;0 " />. Khi đó <img src="https://i.upmath.me/svg/f" alt="f" /> có dạng  
<img src="https://i.upmath.me/svg/f(x)%3D%20%5Cdfrac%7B%5Cmu%7D%7B2%7D%5C%7Cx-b%5C%7C%5E2%20%2Bc%20" alt="f(x)= \dfrac{\mu}{2}\|x-b\|^2 +c " /> với <img src="https://i.upmath.me/svg/b%20%5Cin%20%5Cmathbb%7BR%7D%5Ed%2C%20c%20%5Cin%20%5Cmathbb%7BR%7D" alt="b \in \mathbb{R}^d, c \in \mathbb{R}" />.  
Ta có <img src="https://i.upmath.me/svg/f" alt="f" /> có điểm global minimum duy nhất <img src="https://i.upmath.me/svg/x%5E*" alt="x^*" />.  
SỬ dụng biểu thức phần tích bài ở phần 1 và tính chất của strongly convex function  
<img src="https://i.upmath.me/svg/g_t%5ET(x_t%20-%20x%5E*)%3D%20%5Cnabla%20f(x_t)%5ET%20(x_t%20-%20x%5E*)%20%5Cge%20f(x_t)-f(x%5E*)%20%2B%5Cdfrac%7B%5Cmu%7D%7B2%7D%5C%7Cx_t-x%5E*%5C%7C%5E2" alt="g_t^T(x_t - x^*)= \nabla f(x_t)^T (x_t - x^*) \ge f(x_t)-f(x^*) +\dfrac{\mu}{2}\|x_t-x^*\|^2" />.    
Ta được  
<img src="https://i.upmath.me/svg/f(x_t)-f(x%5E*)%20%5Cle%20%5Cfrac%7B1%7D%7B2%5Ceta%7D(%5Ceta%5E2%5C%7C%5Cnabla%20f(x_t)%5C%7C%5E2%20%2B%5C%7Cx_t%20-%20x%5E*%5C%7C%5E2%20-%5C%7Cx_%7Bt%2B1%7D%20-%20x%5E*%20%5C%7C%5E2)-%20%5Cdfrac%7B%5Cmu%7D%7B2%7D%5C%7Cx_t%20-%20x%5E*%5C%7C%5E2%20%20" alt="f(x_t)-f(x^*) \le \frac{1}{2\eta}(\eta^2\|\nabla f(x_t)\|^2 +\|x_t - x^*\|^2 -\|x_{t+1} - x^* \|^2)- \dfrac{\mu}{2}\|x_t - x^*\|^2  " />  **(1)**  
Suy ra <img src="https://i.upmath.me/svg/%5C%7Cx_%7Bt%2B1%7D%20-%20x%5E*%5C%7C%20%5Cle%202%5Ceta(f(x%5E*)%20-%20f(x_t))%20%2B%5Ceta%5E2%5C%7C%5Cnabla%20f(x_t)%5C%7C%5E2%20%2B(1-%5Cmu%20%5Ceta)%5C%7Cx_t%20-%20x%5E*%5C%7C%5E2" alt="\|x_{t+1} - x^*\| \le 2\eta(f(x^*) - f(x_t)) +\eta^2\|\nabla f(x_t)\|^2 +(1-\mu \eta)\|x_t - x^*\|^2" /> **(2)**  

  
- **Định lý**: Cho hàm <img src="https://i.upmath.me/svg/f%3A%20%5Cmathbb%7BR%7D%5Ed%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: \mathbb{R}^d \rightarrow \mathbb{R}" /> là hàm lồi và khả vi.  Giả sử hàm <img src="https://i.upmath.me/svg/f" alt="f" /> **smooth** với tham số <img src="https://i.upmath.me/svg/L" alt="L" /> và strongly convex với tham số <img src="https://i.upmath.me/svg/%5Cmu" alt="\mu" />. Theo bổ đề 1 ở trên, <img src="https://i.upmath.me/svg/f" alt="f" /> có điểm global minimum duy nhất <img src="https://i.upmath.me/svg/x%5E*" alt="x^*" />. Chọn <img src="https://i.upmath.me/svg/%5Ceta%20%3D%20%5Cdfrac%7B1%7D%7BL%7D" alt="\eta = \dfrac{1}{L}" />.  
gradient descent với điểm bắt đầu <img src="https://i.upmath.me/svg/x_0" alt="x_0" /> bất kì ta có  
**i)**  <img src="https://i.upmath.me/svg/%5C%7Cx_%7Bt%2B1%7D-%20x%5E*%5C%7C%20%5Cle%20(1-%20%5Cdfrac%7B%5Cmu%7D%7BL%7D)%5C%7Cx_t%20-%20x%5E*%5C%7C%5E2" alt="\|x_{t+1}- x^*\| \le (1- \dfrac{\mu}{L})\|x_t - x^*\|^2" />, <img src="https://i.upmath.me/svg/t%20%5Cge%200" alt="t \ge 0" />.  
**ii)** Gọi <img src="https://i.upmath.me/svg/T" alt="T" /> là số vòng lặp tại một thời điểm nào đó.  
<img src="https://i.upmath.me/svg/f(x_T)%20-%20f(x%5E*)%20%5Cle%20%5Cdfrac%7BL%7D%7B2%7D%5Cbigg(1-%20%5Cdfrac%7B%5Cmu%7D%7BL%7D%5Cbigg)%5ET%5C%7Cx_0-%20x%5E*%5C%7C%2C%20T%3E0" alt="f(x_T) - f(x^*) \le \dfrac{L}{2}\bigg(1- \dfrac{\mu}{L}\bigg)^T\|x_0- x^*\|, T&gt;0" />.  
  
**Chứng minh**  
**i)** Sử dụng định lý ở bài **Phần 2** ta có <img src="https://i.upmath.me/svg/f(x%5E*)%20-f(x_t)%20%5Cle%20f(x_%7Bt%2B1%7D)%20-f(x%5E*)%20%5Cle%20-%5Cdfrac%7B1%7D%7B2L%7D%5C%7C%5Cnabla%20f(x_t)%5C%7C%5E2" alt="f(x^*) -f(x_t) \le f(x_{t+1}) -f(x^*) \le -\dfrac{1}{2L}\|\nabla f(x_t)\|^2" />.  
Mà <img src="https://i.upmath.me/svg/%5Ceta%20%3D%5Cdfrac%7B1%7D%7BL%7D" alt="\eta =\dfrac{1}{L}" /> nên ta được <img src="https://i.upmath.me/svg/2%5Ceta%20(f(x%5E*)%20-f%20(x_t))%20%2B%5Ceta%5E2%5C%7C%5Cnabla%20f(x_t)%5C%7C%5E2%20%5Cle%200" alt="2\eta (f(x^*) -f (x_t)) +\eta^2\|\nabla f(x_t)\|^2 \le 0" />.  
Từ **1)** suy ra <img src="https://i.upmath.me/svg/%5C%7Cx_%7Bt%2B1%7D-%20x%5E*%5C%7C%20%5Cle%20(1-%5Cmu%20%5Ceta)%5C%7Cx_t%20-%20x%5E*%5C%7C%5E2%20%5Cle%20%20(1-%20%5Cdfrac%7B%5Cmu%7D%7BL%7D)%5C%7Cx_t%20-%20x%5E*%5C%7C%5E2" alt="\|x_{t+1}- x^*\| \le (1-\mu \eta)\|x_t - x^*\|^2 \le  (1- \dfrac{\mu}{L})\|x_t - x^*\|^2" /> (đpcm).  
**i))** Do <img src="https://i.upmath.me/svg/f" alt="f" /> smooth và <img src="https://i.upmath.me/svg/%5Cnabla%20f(x%5E*)%3D0" alt="\nabla f(x^*)=0" /> nên  
<img src="https://i.upmath.me/svg/f(x_T)%20-f(x%5E*)%20%5Cle%20%5Cnabla%20f(x%5E*)%20(x_T-%20x%5E*)%20%2B%5Cdfrac%7BL%7D%7B2%7D%5C%7Cx_T%20-%20x%5E*%5C%7C%5E2%20%3D%20%5Cdfrac%7BL%7D%7B2%7D%5C%7Cx_T%20-%20x%5E*%5C%7C%5E2" alt="f(x_T) -f(x^*) \le \nabla f(x^*) (x_T- x^*) +\dfrac{L}{2}\|x_T - x^*\|^2 = \dfrac{L}{2}\|x_T - x^*\|^2" />.  
Suy ra đpcm.  
Từ đây cũng suy ra được <img src="https://i.upmath.me/svg/T%20%5Cge%20%5Cdfrac%7BL%7D%7B%5Cmu%7D%20%5Cln%5Cbigg(%5Cdfrac%7BR%5E2L%7D%7B2%5Cepsilon%7D%5Cbigg)" alt="T \ge \dfrac{L}{\mu} \ln\bigg(\dfrac{R^2L}{2\epsilon}\bigg)" />, hay số iteration khoảng <img src="https://i.upmath.me/svg/O%20%5Cbigg(%5Cln%20%5Cbigg(%5Cdfrac%7B1%7D%7B%5Cepsilon%7D%20%5Cbigg)%5Cbigg)" alt="O \bigg(\ln \bigg(\dfrac{1}{\epsilon} \bigg)\bigg)" />.    
Bảng tổng kết  
[![strongly.png](https://i.postimg.cc/JzwGxRhk/strongly.png)](https://postimg.cc/ykP1Y4c1)
#### Tài liệu tham khảo  
1. https://machinelearningcoban.com/
2. https://github.com/epfml/OptML_course
3. https://ee227c.github.io/
4. Stephen Boyd and Lieven Vandenberghe.  
Convex Optimization.  
Cambridge University Press, New York, NY, USA, 2004.  
5. https://easyai.tech/en/ai-definition/gradient-descent/
6. https://towardsdatascience.com/binary-cross-entropy-and-logistic-regression-bf7098e75559?fbclid=IwAR1kSrG7pKJQvmge-M14CUkhjsZ0nlFA1Tw_4tBDWnBkBP8_fblXLrylk3s  
7. http://www.seas.ucla.edu/~vandenbe/236C/lectures/gradient.pdf