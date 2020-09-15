---
layout: post
title: Về tính hội tụ của thuật toán Gradient Descent (Phần 2)
tags: [Convex Optimization]
---





[![B4.png](https://i.postimg.cc/B6N7nH5v/B4.png)](https://postimg.cc/TLLJky48)
- Ở bài trước mình đã phân tính tính hội tụ của Gradient Descent đối với hàm **Lipschitz Convex Function**. Số steps khoảng <img src="https://i.upmath.me/svg/O%5Cbigg(%5Cdfrac%7B1%7D%7B%5Cepsilon%5E2%7D%5Cbigg)" alt="O\bigg(\dfrac{1}{\epsilon^2}\bigg)" />.  
- Bài viết này mình sẽ phân tích tính hội tụ của Gradient Descent với dạng hàm **Smooth Convex Function**.  
(Có thể hình dung hàm này là "Not too curved")  
Nhắc lại tính chất *first-order characterization
of convexity* của hàm lồi:  
Với mọi <img src="https://i.upmath.me/svg/x%2C%20y%20%5Cin%20dom(f)" alt="x, y \in dom(f)" /> ta có <img src="https://i.upmath.me/svg/f(y)%20%5Cge%20f(x)%20%2B%5Cnabla%20f(x)%5ET(y-x)" alt="f(y) \ge f(x) +\nabla f(x)^T(y-x)" />.  
**Định nghĩa [smooth]**:   Cho hàm số <img src="https://i.upmath.me/svg/f%3A%20dom(f)%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: dom(f) \rightarrow \mathbb{R}" /> là hàm khả vi, <img src="https://i.upmath.me/svg/X%20%5Csubseteq%20dom(f)" alt="X \subseteq dom(f)" /> và <img src="https://i.upmath.me/svg/L%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B%2B%7D" alt="L \in \mathbb{R}^{+}" />. Hàm <img src="https://i.upmath.me/svg/f" alt="f" /> được gọi là "trơn" (smooth) (với tham số <img src="https://i.upmath.me/svg/L" alt="L" />) trên <img src="https://i.upmath.me/svg/X" alt="X" /> nếu  
<img src="https://i.upmath.me/svg/f(y)%20%5Cle%20f(x)%20%2B%20%5Cnabla%20f(x)%5ET(y-x)%2B%5Cdfrac%7BL%7D%7B2%7D%5C%7Cx-y%5C%7C%5E2" alt="f(y) \le f(x) + \nabla f(x)^T(y-x)+\dfrac{L}{2}\|x-y\|^2" /> với mọi <img src="https://i.upmath.me/svg/x%2C%20y%20%5Cin%20X" alt="x, y \in X" />.  
**Định nghĩa [smooth convex]**: Cho hàm số <img src="https://i.upmath.me/svg/f%3A%20dom(f)%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: dom(f) \rightarrow \mathbb{R}" /> là hàm lồi, khả vi, <img src="https://i.upmath.me/svg/X%20%5Csubseteq%20dom(f)" alt="X \subseteq dom(f)" /> là tập lồi và <img src="https://i.upmath.me/svg/L%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B%2B%7D" alt="L \in \mathbb{R}^{+}" />. Hàm <img src="https://i.upmath.me/svg/f" alt="f" /> được gọi là "trơn lồi" (smooth convex) (với tham số <img src="https://i.upmath.me/svg/L" alt="L" />) trên <img src="https://i.upmath.me/svg/X" alt="X" /> nếu  
<img src="https://i.upmath.me/svg/f(y)%20%5Cle%20f(x)%20%2B%20%5Cnabla%20f(x)%5ET(y-x)%2B%5Cdfrac%7BL%7D%7B2%7D%5C%7Cx-y%5C%7C%5E2" alt="f(y) \le f(x) + \nabla f(x)^T(y-x)+\dfrac{L}{2}\|x-y\|^2" /> với mọi <img src="https://i.upmath.me/svg/x%2C%20y%20%5Cin%20X" alt="x, y \in X" />.  

- Với trường hợp <img src="https://i.upmath.me/svg/L%3D0" alt="L=0" /> thì ta có <img src="https://i.upmath.me/svg/f(y)%20%3D%20f(x)%20%2B%20%5Cnabla%20f(x)%5ET(y-x)" alt="f(y) = f(x) + \nabla f(x)^T(y-x)" /> (do kết hợp với tính chất *first-order characterization
of convexity* của hàm lồi) suy ra <img src="https://i.upmath.me/svg/f" alt="f" /> là hàm affine.  
Lấy ví dụ với hàm <img src="https://i.upmath.me/svg/f(x)%3Dx%5E2" alt="f(x)=x^2" /> là hàm lồi và dễ thấy rằng <img src="https://i.upmath.me/svg/f(y)%3Dy%5E2%3Dx%5E2%2B2x(y-x)%20%2B%20(x-y)%5E2%3D%20f(x)%20%2Bf'(x)(y-x)%20%2B%5Cdfrac%7BL%7D%7B2%7D(x-y)%5E2" alt="f(y)=y^2=x^2+2x(y-x) + (x-y)^2= f(x) +f'(x)(y-x) +\dfrac{L}{2}(x-y)^2" /> với <img src="https://i.upmath.me/svg/L%3D2" alt="L=2" />. Do vậy hàm <img src="https://i.upmath.me/svg/f(x)%3Dx%5E2" alt="f(x)=x^2" /> là hàm smooth convex với tham số <img src="https://i.upmath.me/svg/L%3D2" alt="L=2" />.  
Hình sau minh họa hàm smooth convex.  
[![B4.png](https://i.postimg.cc/RFQ1J0W7/B4.png)](https://postimg.cc/DJ04tnd0)
  
Tổng quát hơn xét hàm bậc hai một biến <img src="https://i.upmath.me/svg/f(x)%3Dax%5E2%2Bbx%2Bc" alt="f(x)=ax^2+bx+c" /> với <img src="https://i.upmath.me/svg/a%3E0" alt="a&gt;0" /> cũng là một hàm smooth convex với <img src="https://i.upmath.me/svg/L%3D2a" alt="L=2a" />.  
Có thể tự kiểm chứng với <img src="https://i.upmath.me/svg/f(y)%20%3D%20f(x)%20%2Bf'(x)(y-x)%2B%5Cdfrac%7BL%7D%7B2%7D(x-y)%5E2" alt="f(y) = f(x) +f'(x)(y-x)+\dfrac{L}{2}(x-y)^2" /> với <img src="https://i.upmath.me/svg/L%3D2a" alt="L=2a" />.  
Còn với biến <img src="https://i.upmath.me/svg/x" alt="x" /> là một vector, **dạng toàn phương (quadratic form)** có dạng <img src="https://i.upmath.me/svg/f(x)%3Dx%5ETAx%2Bb%5ETx%2Bc" alt="f(x)=x^TAx+b^Tx+c" /> với <img src="https://i.upmath.me/svg/A" alt="A" /> là ma trận đối xứng và nửa xác định dương thì hàm này có phải là hàm smooth convex không? Câu trả lời là có.  
Ta có định lý sau:  
**Định lý**: Cho hàm <img src="https://i.upmath.me/svg/f(x)%3D%20x%5ETQx%2Bb%5ETx%2Bc" alt="f(x)= x^TQx+b^Tx+c" /> trong đó <img src="https://i.upmath.me/svg/Q" alt="Q" /> là ma trận nửa xác định dương, đối xứng, <img src="https://i.upmath.me/svg/Q%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%20%5Ctimes%20d%7D%2C%20b%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%7D%2C%20c%20%5Cin%20%5Cmathbb%7BR%7D" alt="Q \in \mathbb{R}^{d \times d}, b \in \mathbb{R}^{d}, c \in \mathbb{R}" />. Khi đó <img src="https://i.upmath.me/svg/f" alt="f" /> smooth convex với tham số <img src="https://i.upmath.me/svg/2%5C%7CQ%5C%7C" alt="2\|Q\|" />, trong đó <img src="https://i.upmath.me/svg/%5C%7CQ%5C%7C" alt="\|Q\|" /> là spectral norm của <img src="https://i.upmath.me/svg/Q" alt="Q" />.  
Về spectral norm của ma trận được định nghĩa như sau:  
**Định nghĩa [spectral norm of matrix]**. Cho ma trận <img src="https://i.upmath.me/svg/A%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bm%20%5Ctimes%20n%7D" alt="A \in \mathbb{R}^{m \times n}" />. Khi đó  
<img src="https://i.upmath.me/svg/%5C%7CA%5C%7C%3A%3D%20%5Cmax_%7Bv%20%5Cin%20%5Cmathbb%7BR%7D%5En%2C%20v%20%5Cne%200%7D%20%5Cdfrac%7B%5C%7CAv%5C%7C%7D%7B%5C%7Cv%5C%7C%7D%3D%20%5Cmax_%7B%5C%7Cv%5C%7C%3D1%7D%5C%7CAv%5C%7C" alt="\|A\|:= \max_{v \in \mathbb{R}^n, v \ne 0} \dfrac{\|Av\|}{\|v\|}= \max_{\|v\|=1}\|Av\|" />  
được gọi là <img src="https://i.upmath.me/svg/2-norm" alt="2-norm" /> hay **spectral norm** của <img src="https://i.upmath.me/svg/A" alt="A" />.  
Lưu ý:  spectral norm này khác với Frobenius norm của ma trận.  
**Chứng minh**  
Do <img src="https://i.upmath.me/svg/Q" alt="Q" /> là ma trận đối xứng, nửa xác định dương nên <img src="https://i.upmath.me/svg/f(x)" alt="f(x)" /> là hàm lồi.  Hơn nữa ta cũng có <img src="https://i.upmath.me/svg/x%5ETQy%3Dy%5ETQx" alt="x^TQy=y^TQx" /> với <img src="https://i.upmath.me/svg/x%2C%20y" alt="x, y" /> bất kỳ.  Khi đó có thể viết lại  
<img src="https://i.upmath.me/svg/f(y)%3Dy%5ETQy%3Dx%5ETQx%2B2x%5ETQ(y-x)%2B(x-y)%5ETQ(x-y)%3D%20%5C%5C%20%20f(x)%2B%202x%5ETQ(y-x)%2B(x-y)%5ETQ(x-y)" alt="f(y)=y^TQy=x^TQx+2x^TQ(y-x)+(x-y)^TQ(x-y)= \\  f(x)+ 2x^TQ(y-x)+(x-y)^TQ(x-y)" />  

Sử dụng bất đẳng thức Cauchy-Schwarz và tính chất của spectral norm ta có  
<img src="https://i.upmath.me/svg/(x-y)%5ETQ(x-y)%20%5Cle%20%5C%7Cx-y%5C%7C%5C%7CQ(x-y)%5C%7C%5Cle%20%5C%7Cx-y%5C%7C%5C%7CQ%5C%7C%5C%7Cx-y%5C%7C%3D%5C%7CQ%5C%7C%5C%7Cx-y%5C%7C%5E2" alt="(x-y)^TQ(x-y) \le \|x-y\|\|Q(x-y)\|\le \|x-y\|\|Q\|\|x-y\|=\|Q\|\|x-y\|^2" />  
Suy ra <img src="https://i.upmath.me/svg/f(y)%20%5Cle%20f(x)%20%2B2x%5ETQ(y-x)%20%2B%20%5C%7CQ%5C%7C%5C%7Cx-y%5C%7C%5E2" alt="f(y) \le f(x) +2x^TQ(y-x) + \|Q\|\|x-y\|^2" />. Mà <img src="https://i.upmath.me/svg/%5Cnabla%20f(x)%20%3D2x%5ETQ" alt="\nabla f(x) =2x^TQ" />. Do đó  
<img src="https://i.upmath.me/svg/f(y)%20%5Cle%20f(x)%20%2B%20%5Cnabla%20f(x)(y-x)%20%2B%5Cdfrac%7B2%5C%7CQ%5C%7C%7D%7B2%7D%5C%7Cx-y%5C%7C%5E2" alt="f(y) \le f(x) + \nabla f(x)(y-x) +\dfrac{2\|Q\|}{2}\|x-y\|^2" />. Điều này chứng tỏ <img src="https://i.upmath.me/svg/f" alt="f" /> smooth với tham số <img src="https://i.upmath.me/svg/2%5C%7CQ%5C%7C" alt="2\|Q\|" />  
  
Tiếp theo là một định lý về hàm smooth convex.  
**Định lý**: Cho <img src="https://i.upmath.me/svg/f%3A%20%5Cmathbb%7BR%7D%5Ed%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: \mathbb{R}^d \rightarrow \mathbb{R}" /> là hàm lồi, khả vi. Hai mệnh đề sau là tương đương.  
**i)** <img src="https://i.upmath.me/svg/f" alt="f" /> smooth với tham số <img src="https://i.upmath.me/svg/L" alt="L" />.  
**ii)** <img src="https://i.upmath.me/svg/%5C%7C%5Cnabla%20f(x)-%20%5Cnabla%20f(y)%7C%20%5Cle%20L%5C%7Cx-y%5C%7C" alt="\|\nabla f(x)- \nabla f(y)| \le L\|x-y\|" /> với mọi <img src="https://i.upmath.me/svg/x%2C%20y%20%5Cin%20%5Cmathbb%7BR%7D%5Ed" alt="x, y \in \mathbb{R}^d" />.  
**Chứng minh**:  
Sử dụng tính chất của tập lồi, xét <img src="https://i.upmath.me/svg/x%2C%20y%20%5Cin%20dom(f)" alt="x, y \in dom(f)" /> thì <img src="https://i.upmath.me/svg/ty%2B(1-t)x%20%3D%20x%20%2Bt(y-x)" alt="ty+(1-t)x = x +t(y-x)" /> cũng thuộc <img src="https://i.upmath.me/svg/dom(f)" alt="dom(f)" /> với <img src="https://i.upmath.me/svg/t%20%5Cin%20%5B0%2C1%5D" alt="t \in [0,1]" />.  
Xét hàm <img src="https://i.upmath.me/svg/g(t)%3Df(x%2Bt(y-x)))" alt="g(t)=f(x+t(y-x)))" />. Giả sử ta có **ii)** thì ta có:  
<img src="https://i.upmath.me/svg/g'(t)%20-%20g'(0)%3D%20(%5Cnabla%20f(x%2Bt(y-x))-%20%5Cnabla%20f(x))%5ET(y-x)%20%5Cle%20tL%5C%7Cx-y%5C%7C%5E2" alt="g'(t) - g'(0)= (\nabla f(x+t(y-x))- \nabla f(x))^T(y-x) \le tL\|x-y\|^2" />  
Lấy tích phân từ <img src="https://i.upmath.me/svg/t%3D0" alt="t=0" /> đến <img src="https://i.upmath.me/svg/t%3D1" alt="t=1" /> ta có:  
<img src="https://i.upmath.me/svg/f(y)%20%3D%20g(1)%20%3D%20g(0)%2B%20%5Cdisplaystyle%20%5Cint_%7B0%7D%5E1%20g'(t)dt%20%5Cle%20g'(0)%20%2B%20%5Cdfrac%7BL%7D%7B2%7D%5C%7Cx-y%5C%7C%5E2%20%3D%20%20%5C%5C%0Af(x)%20%2B%20%5Cnabla%20f(x)%5ET(y-x)%20%2B%20%5Cdfrac%7BL%7D%7B2%7D%5C%7Cx-y%5C%7C%5E2" alt="f(y) = g(1) = g(0)+ \displaystyle \int_{0}^1 g'(t)dt \le g'(0) + \dfrac{L}{2}\|x-y\|^2 =  \\
f(x) + \nabla f(x)^T(y-x) + \dfrac{L}{2}\|x-y\|^2" />.  
Từ đây suy ra <img src="https://i.upmath.me/svg/f" alt="f" /> smooth convex với tham số <img src="https://i.upmath.me/svg/L" alt="L" />.  
Ngược lại nếu ta có **i)** tức là <img src="https://i.upmath.me/svg/f(y)%20%5Cle%20f(x)%20%2B%20%5Cnabla%20f(x)%5ET(y-x)%20%2B%20%5Cdfrac%7BL%7D%7B2%7D%5C%7Cx-y%5C%7C%5E2" alt="f(y) \le f(x) + \nabla f(x)^T(y-x) + \dfrac{L}{2}\|x-y\|^2" />.  
Hoán đổi <img src="https://i.upmath.me/svg/x%2C%20y" alt="x, y" /> ta được <img src="https://i.upmath.me/svg/f(x)%20%5Cle%20f(y)%20%2B%20%5Cnabla%20f(y)%5ET(x-y)%20%2B%20%5Cdfrac%7BL%7D%7B2%7D%5C%7Cy-x%5C%7C%5E2" alt="f(x) \le f(y) + \nabla f(y)^T(x-y) + \dfrac{L}{2}\|y-x\|^2" />.  
Từ đây dễ dàng suy ra <img src="https://i.upmath.me/svg/%5C%7C%5Cnabla%20f(x)-%20%5Cnabla%20f(y)%7C%20%5Cle%20L%5C%7Cx-y%5C%7C" alt="\|\nabla f(x)- \nabla f(y)| \le L\|x-y\|" />.  
Như vậy **i)** và **i))** tương đương nhau.  
**Định lý**: Cho hàm <img src="https://i.upmath.me/svg/f%3A%20%5Cmathbb%7BR%7D%5Ed%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: \mathbb{R}^d \rightarrow \mathbb{R}" /> khả vi và trơn với tham số <img src="https://i.upmath.me/svg/L" alt="L" />. Với learning rate <img src="https://i.upmath.me/svg/%5Ceta%20%3D%5Cdfrac%7B1%7D%7BL%7D" alt="\eta =\dfrac{1}{L}" /> khi áp dụng thuật toán gradient descent ta có  
<img src="https://i.upmath.me/svg/f(x_%7Bt%2B1%7D)%20%5Cle%20f(x_t)-%5Cdfrac%7B1%7D%7B2L%7D%5C%7C%5Cnabla%20f(x_t)%5C%7C%5E2" alt="f(x_{t+1}) \le f(x_t)-\dfrac{1}{2L}\|\nabla f(x_t)\|^2" />, <img src="https://i.upmath.me/svg/t%20%5Cge%200" alt="t \ge 0" />  
**Chứng minh**  
Với learning rate <img src="https://i.upmath.me/svg/%5Ceta%20%3D%5Cdfrac%7B1%7D%7BL%7D" alt="\eta =\dfrac{1}{L}" /> thì theo công thức cập nhật của gradient descent ta có <img src="https://i.upmath.me/svg/x_%7Bt%2B1%7D-x_t%3D%20-%5Cdfrac%7B%5Cnabla%20f(x_t)%7D%7BL%7D" alt="x_{t+1}-x_t= -\dfrac{\nabla f(x_t)}{L}" /> và áp dụng công thức hàm trơn:  
<img src="https://i.upmath.me/svg/f(x_%7Bt%2B1%7D)%20%5Cle%20f(x_t)%20%2B%20%5Cnabla%20f(x_t)%5ET(x_%7Bt%2B1%7D-x_t)%20%2B%20%5Cdfrac%7BL%7D%7B2%7D%5C%7Cx_t%20-x_%7Bt%2B1%7D%5C%7C%5E2%20%5C%5C%0A%3D%20f(x_t)%20-%5Cdfrac%7B1%7D%7BL%7D%5C%7C%5Cnabla%20f(x_t)%5C%7C%5E2%2B%5Cdfrac%7B1%7D%7B2L%7D%5C%7C%5Cnabla%20f(x_t)%5C%7C%5E2%5C%5C%0A%3D%20f(x_t)%20-%20%5Cdfrac%7B1%7D%7B2L%7D%5C%7C%5Cnabla%20f(x_t)%5C%7C%5E2" alt="f(x_{t+1}) \le f(x_t) + \nabla f(x_t)^T(x_{t+1}-x_t) + \dfrac{L}{2}\|x_t -x_{t+1}\|^2 \\
= f(x_t) -\dfrac{1}{L}\|\nabla f(x_t)\|^2+\dfrac{1}{2L}\|\nabla f(x_t)\|^2\\
= f(x_t) - \dfrac{1}{2L}\|\nabla f(x_t)\|^2" />  
Sau đây là định lý về sự hội tụ của gradient descent với hàm smooth convex.  
**Định lý**: Cho hàm <img src="https://i.upmath.me/svg/f%3A%20%5Cmathbb%7BR%7D%5Ed%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: \mathbb{R}^d \rightarrow \mathbb{R}" /> lồi, khả vi với điểm **global minimum** là <img src="https://i.upmath.me/svg/x%5E*" alt="x^*" />, giả sử <img src="https://i.upmath.me/svg/f" alt="f" /> trơn với tham số <img src="https://i.upmath.me/svg/L" alt="L" />. Với learning rate <img src="https://i.upmath.me/svg/%5Ceta%20%3D%20%5Cdfrac%7B1%7D%7BL%7D" alt="\eta = \dfrac{1}{L}" /> khi áp dụng gradient descent ta có:  
<img src="https://i.upmath.me/svg/f(x_T)-f(x%5E*)%20%5Cle%20%5Cdfrac%7BL%7D%7B2T%7D%5C%7Cx_0%20-%20x%5E*%5C%7C%5E2" alt="f(x_T)-f(x^*) \le \dfrac{L}{2T}\|x_0 - x^*\|^2" />, <img src="https://i.upmath.me/svg/T%3E0" alt="T&gt;0" />  
**Chứng minh**  
Áp dụng định lý trên và lấy tổng từ <img src="https://i.upmath.me/svg/0%20%5Crightarrow%20T-1%20" alt="0 \rightarrow T-1 " /> ta có  
<img src="https://i.upmath.me/svg/%5Cdisplaystyle%20%5Cdfrac%7B1%7D%7B2L%7D%20%5Csum_%7Bt%3D0%7D%5E%7BT-1%7D%5C%7C%5Cnabla%20f(x_t)%5C%7C%5E2%20%5Cle%20%5Csum_%7Bt%3D0%7D%5E%7BT-1%7D%20(f(x_t)%20-%20f(x_%7Bt%2B1%7D))%20%3D%20f(x_0)%20-f(x_T)%20" alt="\displaystyle \dfrac{1}{2L} \sum_{t=0}^{T-1}\|\nabla f(x_t)\|^2 \le \sum_{t=0}^{T-1} (f(x_t) - f(x_{t+1})) = f(x_0) -f(x_T) " />.  
Với <img src="https://i.upmath.me/svg/%5Ceta%20%3D%20%5Cdfrac%7B1%7D%7BL%7D" alt="\eta = \dfrac{1}{L}" /> và áp dụng phân tích đã có trong bài [Về tính hội tụ của thuật toán Gradient Descent (Phần 1)](https://anhquannguyen21.github.io./2020-09-10-V%E1%BB%81-t%C3%ADnh-h%E1%BB%99i-t%E1%BB%A5-c%E1%BB%A7a-thu%E1%BA%ADt-to%C3%A1n-Gradient-Descent-(Ph%E1%BA%A7n-1)/) ta có:  
<img src="https://i.upmath.me/svg/%5Cdisplaystyle%20%5Csum_%7Bt%3D0%7D%5E%7BT-1%7D(f(x_t)%20-%20f(x%5E*))%20%5Cle%20%5Cdfrac%7B1%7D%7B2L%7D%5Csum_%7Bt%3D0%7D%5E%7BT-1%7D%5C%7C%5Cnabla%20f(x_t)%5C%7C%5E2%20%2B%5Cdfrac%7BL%7D%7B2%7D%5C%7Cx_0%20-%20x%5E*%5C%7C%20" alt="\displaystyle \sum_{t=0}^{T-1}(f(x_t) - f(x^*)) \le \dfrac{1}{2L}\sum_{t=0}^{T-1}\|\nabla f(x_t)\|^2 +\dfrac{L}{2}\|x_0 - x^*\| " /> hay  
<img src="https://i.upmath.me/svg/%5Cdisplaystyle%20%5Csum_%7Bt%3D1%7D%5E%7BT%7D(f(x_t)%20-%20f(x%5E*))%20%5Cle%20%5Cdfrac%7BL%7D%7B2%7D%5C%7Cx_0%20-%20x%5E*%5C%7C" alt="\displaystyle \sum_{t=1}^{T}(f(x_t) - f(x^*)) \le \dfrac{L}{2}\|x_0 - x^*\|" />.  
Mặc khác do <img src="https://i.upmath.me/svg/f(x_%7Bt%2B1%7D)%20%5Cle%20f(x_t)" alt="f(x_{t+1}) \le f(x_t)" /> với <img src="https://i.upmath.me/svg/0%20%5Cle%20t%20%5Cle%20T" alt="0 \le t \le T" /> nên  
<img src="https://i.upmath.me/svg/f(x_T)%20-f(x%5E*)%5Cle%20%5Cdfrac%7B1%7D%7BT%7D%20%5Csum_%7Bt%3D1%7D%5E%7BT%7D(f(x_t)%20-%20f(x%5E*))%20%5Cle%20%20%5Cdfrac%7BL%7D%7B2T%7D%5C%7Cx_0%20-%20x%5E*%5C%7C%5E2" alt="f(x_T) -f(x^*)\le \dfrac{1}{T} \sum_{t=1}^{T}(f(x_t) - f(x^*)) \le  \dfrac{L}{2T}\|x_0 - x^*\|^2" />.  
Giả sử khoảng cách giữa điểm khởi tạo và điểm global minimum là <img src="https://i.upmath.me/svg/R" alt="R" />, tức là <img src="https://i.upmath.me/svg/%5C%7Cx_0%20-%20x%5E*%5C%7C%20%3DR" alt="\|x_0 - x^*\| =R" />. Khi đó ta cần <img src="https://i.upmath.me/svg/T%20%5Cge%20%5Cdfrac%7BR%5E2L%7D%7B2%5Cepsilon%7D" alt="T \ge \dfrac{R^2L}{2\epsilon}" /> để khoảng cách error là <img src="https://i.upmath.me/svg/%5Cepsilon" alt="\epsilon" />.  
Ta thấy số vòng lặp ít nhất phụ thuộc vào cả <img src="https://i.upmath.me/svg/R" alt="R" /> và <img src="https://i.upmath.me/svg/%5Cepsilon" alt="\epsilon" />.  
Từ đây ta ước lượng số vòng lặp với hàm smooth convex khoảng <img src="https://i.upmath.me/svg/O%5Cbigg(%5Cdfrac%7B1%7D%7B%5Cepsilon%7D%5Cbigg)" alt="O\bigg(\dfrac{1}{\epsilon}\bigg)" />.  Nhanh hơn so với hàm Lipschitz convex.  
Đặc biệt với hàm <img src="https://i.upmath.me/svg/f(x)%3Dx%5E2" alt="f(x)=x^2" /> hay tổng quát hơn <img src="https://i.upmath.me/svg/f(x)%3Dax%5E2%2Bbx%2Bc" alt="f(x)=ax^2+bx+c" /> là hàm smooth convex với tham số <img src="https://i.upmath.me/svg/L%3D2a" alt="L=2a" /> nên nếu chọn learning rate là <img src="https://i.upmath.me/svg/%5Ceta%20%3D%5Cdfrac%7B1%7D%7BL%7D" alt="\eta =\dfrac{1}{L}" /> thì chỉ cần sau một vòng lặp thì gradient descent đã hội tụ đúng điểm tối ưu.  Vì khi đó với <img src="https://i.upmath.me/svg/t%5Cge%201" alt="t\ge 1" /> thì <img src="https://i.upmath.me/svg/f(x_t)%3Df(x_%7Bt%2B1%7D)%3D%20f(x%5E*)" alt="f(x_t)=f(x_{t+1})= f(x^*)" />.  
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
  





  



 
