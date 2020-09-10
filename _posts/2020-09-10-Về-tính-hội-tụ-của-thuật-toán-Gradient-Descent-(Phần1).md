---
layout: post
title: Về tính hội tụ của thuật toán Gradient Descent (Phần 1)
tags: [Convex Optimization]
---

[![B4.png](https://i.postimg.cc/B6N7nH5v/B4.png)](https://postimg.cc/TLLJky48)
- Trong bài toán tối ưu cụ thể là các bài toán tìm giá trị lớn nhất hay giá trị nhỏ nhất của một hàm số, ta thường đi giải phương trình đạo hàm hoặc đạo hàm riêng bằng 0 đối với hàm nhiều biến. Tuy nhiên, việc tối ưu hàm lỗi trong Machine Learning gặp phải các vấn đề sau:  
  - Với những tập dữ liệu lớn, nhiều chiều thì sẽ gây ra tốn bộ nhớ của máy tính và tính toán chậm chạp. Ví dụ công thức tính tham số của thuật toán Linear Regression như sau <img src="https://i.upmath.me/svg/w%3D(X%5E%7BT%7DX)%5E%7B%5Cdagger%7DX%5E%7BT%7Dy" alt="w=(X^{T}X)^{\dagger}X^{T}y" />  (kí hiệu <img src="https://i.upmath.me/svg/%20A%5E%7B%5Cdagger%7D" alt=" A^{\dagger}" /> là ma trận giả nghịch đảo). Khi đó nếu dữ liệu train lớn và chiều của dữ liệu lớn thì kích thước ma trận <img src="https://i.upmath.me/svg/X" alt="X" /> sẽ lớn, việc tính nghịch đảo và nhân ma trận sẽ chậm.
  - Trong nhiều thuật toán Machine Learning các hàm lỗi (Loss Function) đôi khi rất phức tạp nên việc tính đạo hàm là không khả thi.  
Và thuật toán Gradient Descent sẽ giải quyết hai vấn đề nêu trên.
- Ý tưởng của Gradient Descent là giá trị hàm số sẽ giảm nhanh nhất khi đi ngược hướng đạo hàm.  
Công thức cập nhật trọng số:  <img src="https://i.upmath.me/svg/%5Ctheta%20%3D%20%5Ctheta%20-%20%5Ceta%20%5Cnabla_%7B%5Ctheta%7D%20f(%5Ctheta)" alt="\theta = \theta - \eta \nabla_{\theta} f(\theta)" /> với <img src="https://i.upmath.me/svg/%5Ctheta%20" alt="\theta " /> là tham số mô hình và <img src="https://i.upmath.me/svg/%5Ceta%20" alt="\eta " /> gọi là tốc độ học (learning rate).  
Chi tiết rõ hơn về thuật toán, các biến thể và cách cài đặt có thể xem tại  [Blog Machine Learning cơ bản](https://machinelearningcoban.com/2017/01/12/gradientdescent/)
- Ở bài này mình sẽ phân tích tính hội tụ của Gradient Descent, số lần lặp tối thiểu để thuật toán hội tụ và tốc độ hội tụ của từng dạng hàm số (các hàm được đề cập đều có tính chất chung là hàm lồi). Trước tiên ta đi qua một số kiến thức chuẩn bị.  
### I. Một số kiến thức chuẩn bị  
#### 1. Lý thuyết về tập lồi, hàm lồi
- Một tập <img src="https://i.upmath.me/svg/C" alt="C" /> được gọi là tập lồi nếu mọi điểm trên đoạn thẳng nối 2 điểm bất kỳ trong tập <img src="https://i.upmath.me/svg/C" alt="C" /> đều thuộc tập hợp <img src="https://i.upmath.me/svg/C" alt="C" />. Tức là, với 2 điểm bất kỳ <img src="https://i.upmath.me/svg/x%2C%20y%20%5Cin%20C" alt="x, y \in C" /> và với <img src="https://i.upmath.me/svg/0%20%5Cle%20%5Clambda%20%5Cle%201" alt="0 \le \lambda \le 1" /> ta có <img src="https://i.upmath.me/svg/%5Clambda%20x%20%2B%20(1-%5Clambda)y%20%5Cin%20C" alt="\lambda x + (1-\lambda)y \in C" />.  
- Ví dụ tập lồi  
[![B4.png](https://i.postimg.cc/c1D7Fbg5/B4.png)](https://postimg.cc/mhHFDVSQ)
- Ví dụ tập không lồi  
[![B42.png](https://i.postimg.cc/sXLYgBrC/B42.png)](https://postimg.cc/TLrLk31t)
- Một hàm <img src="https://i.upmath.me/svg/f%3A%20%5Cmathbb%7BR%7D%5E%7Bd%7D%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: \mathbb{R}^{d} \rightarrow \mathbb{R}" /> được gọi là hàm lồi nếu miền xác định của <img src="https://i.upmath.me/svg/f" alt="f" />, <img src="https://i.upmath.me/svg/dom(f)" alt="dom(f)" /> là một tập lồi và với mọi <img src="https://i.upmath.me/svg/x%2C%20y%20%5Cin%20dom(f)" alt="x, y \in dom(f)" /> và <img src="https://i.upmath.me/svg/%5Clamba%2C%200%20%5Cle%20%5Clambda%20%5Cle%201" alt="\lamba, 0 \le \lambda \le 1" /> ta có  
<img src="https://i.upmath.me/svg/f(%5Clambda%20x%20%2B(1-%20%5Clambda)%20y)%20%5Cle%20%5Clambda%20f(x)%20%2B%20(1-%20%5Clambda)f(y)" alt="f(\lambda x +(1- \lambda) y) \le \lambda f(x) + (1- \lambda)f(y)" />.  
Một cách hình học: Đoạn thẳng nối giữa <img src="https://i.upmath.me/svg/(x%2C%20f(x))%2C%20(y%2C%20f(y))" alt="(x, f(x)), (y, f(y))" /> nằm trên đồ thị của <img src="https://i.upmath.me/svg/f" alt="f" />.  
[![B4.png](https://i.postimg.cc/02H7GWXd/B4.png)](https://postimg.cc/7f0CwVkf)
- [Hàm lồi chặt - Strictly Convex Function]. Hàm <img src="https://i.upmath.me/svg/f%3A%20dom(f)%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: dom(f) \rightarrow \mathbb{R}" /> là hàm lồi chặt nếu <img src="https://i.upmath.me/svg/dom(f)" alt="dom(f)" /> lồi và với mọi <img src="https://i.upmath.me/svg/x%20%5Cne%20y%20%5Cin%20dom(f)" alt="x \ne y \in dom(f)" /> và với mọi <img src="https://i.upmath.me/svg/%5Clambda%20%5Cin%20(0%2C%201)" alt="\lambda \in (0, 1)" />, ta có  
<img src="https://i.upmath.me/svg/f(%5Clambda%20x%20%2B(1-%20%5Clambda)%20y)%20%3C%20%5Clambda%20f(x)%20%2B%20(1-%20%5Clambda)f(y)" alt="f(\lambda x +(1- \lambda) y) &lt; \lambda f(x) + (1- \lambda)f(y)" />
- Một số hàm lồi
   - Linear functions: <img src="https://i.upmath.me/svg/f(x)%3Da%5ETx" alt="f(x)=a^Tx" />  
   - Affine functions: <img src="https://i.upmath.me/svg/f(x)%3Da%5ETx%2Bb" alt="f(x)=a^Tx+b" />
   - Exponential: <img src="https://i.upmath.me/svg/f(x)%3De%5E%7B%5Calpha%20x%7D" alt="f(x)=e^{\alpha x}" />  
   - Norms: Mọi norm trên <img src="https://i.upmath.me/svg/%5Cmathbb%7BR%7D%5Ed" alt="\mathbb{R}^d" /> đều là hàm lồi.  
     Ví dụ xét tính lồi của <img src="https://i.upmath.me/svg/%5C%7Cx%5C%7C" alt="\|x\|" />.
     Sử dụng tính chất của norm là bất đẳng thức tam giác <img src="https://i.upmath.me/svg/%5C%7Cx%2By%5C%7C%20%5Cle%20%5C%7Cx%5C%7C%20%2B%20%5C%7Cy%5C%7C" alt="\|x+y\| \le \|x\| + \|y\|" /> và <img src="https://i.upmath.me/svg/%5C%7Cax%5C%7C%20%3Da%5C%7Cx%5C%7C" alt="\|ax\| =a\|x\|" /> với <img src="https://i.upmath.me/svg/a" alt="a" /> là số thực.  
     Ta có <img src="https://i.upmath.me/svg/%5C%7C%20%5Clambda%20x%2B%20(1-%5Clambda)y%5C%7C%20%5Cle%20%5C%7C%5Clambda%20x%5C%7C%2B%20%5C%7C(1-%5Clambda)%20y%5C%7C%3D%20%5Clambda%20%5C%7Cx%5C%7C%2B%20(1-%5Clambda)%5C%7Cy%5C%7C" alt="\| \lambda x+ (1-\lambda)y\| \le \|\lambda x\|+ \|(1-\lambda) y\|= \lambda \|x\|+ (1-\lambda)\|y\|" />. Do vậy norm là một hàm lồi.  
#### 2. Kiểm tra tính chất lồi dựa vào đạo hàm bậc nhất (First-order characterization of convexity)
- Định lý: Giả sử hàm số <img src="https://i.upmath.me/svg/f" alt="f" /> có tập xác định <img src="https://i.upmath.me/svg/dom(f)" alt="dom(f)" />, có đạo hàm tại mọi điểm trên tập xác định, hay vector gradient của <img src="https://i.upmath.me/svg/f" alt="f" /> là <img src="https://i.upmath.me/svg/%5Cnabla%20f%20%3A%3D%20%5Cbigg(%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_1%7D(x)%2C...%2C%20%5Cdfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_d%7D(x)%5Cbigg)" alt="\nabla f := \bigg(\dfrac{\partial f}{\partial x_1}(x),..., \dfrac{\partial f}{\partial x_d}(x)\bigg)" /> tồn tại với mọi <img src="https://i.upmath.me/svg/x%20%5Cin%20dom(f)" alt="x \in dom(f)" />. Khi đó hàm <img src="https://i.upmath.me/svg/f" alt="f" /> lồi nếu và chỉ nếu <img src="https://i.upmath.me/svg/dom(f)" alt="dom(f)" /> lồi và <img src="https://i.upmath.me/svg/f(y)%20%5Cge%20f(x)%2B%20%5Cnabla%20f(x)%5ET(y-x)" alt="f(y) \ge f(x)+ \nabla f(x)^T(y-x)" /> với mọi <img src="https://i.upmath.me/svg/x%2C%20y%20%5Cin%20dom(f)" alt="x, y \in dom(f)" />.  
Một cách trực quan: Hàm số là lồi nếu mặt tiếp tuyến tại một điểm bất kỳ trên đồ thị hàm số không nằm trên đồ thị đó.
 [![B4.png](https://i.postimg.cc/8z62FM06/B4.png)](https://postimg.cc/bZqCWD8y)
#### 3. Kiểm tra tính chất lồi dựa vào đạo hàm bậc hai (Second-order characterization of convexity)
- Hàm <img src="https://i.upmath.me/svg/f%3A%20dom(f)%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: dom(f) \rightarrow \mathbb{R}" /> được gọi là twice continuously differentiable nếu <img src="https://i.upmath.me/svg/%5Cnabla%20f" alt="\nabla f" /> có đạo hàm (differentiable) và <img src="https://i.upmath.me/svg/%5Cnabla%5E2%20f" alt="\nabla^2 f" /> liên tục (continuous).
- Định lý: Giả sử hàm số <img src="https://i.upmath.me/svg/f" alt="f" /> có tập xác định <img src="https://i.upmath.me/svg/dom(f)" alt="dom(f)" />, và <img src="https://i.upmath.me/svg/f" alt="f" /> twice
continuously differentiable và ma trận Hessian <img src="https://i.upmath.me/svg/H" alt="H" /> của <img src="https://i.upmath.me/svg/f" alt="f" />  
[![B4.png](https://i.postimg.cc/fbhqPFTH/B4.png)](https://postimg.cc/fVBvt8dX)  
 là ma trận đối xứng và tồn tại với mọi <img src="https://i.upmath.me/svg/x%5Cin%20dom(f)" alt="x\in dom(f)" />. Khi đó <img src="https://i.upmath.me/svg/f" alt="f" /> là hàm lồi nếu và chỉ nếu <img src="https://i.upmath.me/svg/dom(f)" alt="dom(f)" /> lồi và với mọi <img src="https://i.upmath.me/svg/x%20%5Cin%20dom(f)" alt="x \in dom(f)" /> ta có <img src="https://i.upmath.me/svg/%20H%20%5Csucceq0" alt=" H \succeq0" />, hay <img src="https://i.upmath.me/svg/H" alt="H" /> là ma trận nửa xác định dương. Nếu <img src="https://i.upmath.me/svg/f" alt="f" /> là hàm lồi chặt thì <img src="https://i.upmath.me/svg/H" alt="H" /> là ma trận xác định dương.  
#### 4. Local Minimum và Global Minimum
- Vấn đề của Gradient Descent là nó chỉ tìm được một điểm cực tiểu local minimum và ta không biết được điểm đó có phải là global minimum hay không. Tuy nhiên có một tính chất qua trọng sau: Nếu hàm số đó làm hàm lồi (convex) thì local minimum đó cũng chính là global minimum, tính chất này sẽ chứng minh sau đây.  
- Định nghĩa: Một local minimum (điểm tối ưu cục bộ) của hàm <img src="https://i.upmath.me/svg/f%3A%20dom(f)%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: dom(f) \rightarrow \mathbb{R}" /> là điểm <img src="https://i.upmath.me/svg/x" alt="x" /> sao cho tồn tại <img src="https://i.upmath.me/svg/%5Cepsilon%20%3E%200" alt="\epsilon &gt; 0" /> với <img src="https://i.upmath.me/svg/f(x)%20%5Cle%20f(y)" alt="f(x) \le f(y)" /> với mọi <img src="https://i.upmath.me/svg/y%20%5Cin%20dom(f)" alt="y \in dom(f)" /> thỏa mãn <img src="https://i.upmath.me/svg/%5C%7Cy-x%5C%7C%20%3C%20%5Cepsilon" alt="\|y-x\| &lt; \epsilon" /> .  
- Định lý: Gọi <img src="https://i.upmath.me/svg/x%5E*" alt="x^*" /> là local minimum của hàm lồi <img src="https://i.upmath.me/svg/f%3A%20dom(f)%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: dom(f) \rightarrow \mathbb{R}" />. Khi đó <img src="https://i.upmath.me/svg/x%5E*" alt="x^*" /> cũng là global minimum (tối ưu toàn cục), nghĩa là <img src="https://i.upmath.me/svg/f(x%5E*)%20%5Cle%20f(y)" alt="f(x^*) \le f(y)" /> với mọi <img src="https://i.upmath.me/svg/y%20%5Cin%20dom(f)" alt="y \in dom(f)" />.  
**Chứng minh:** Giả sử rằng tồn tại <img src="https://i.upmath.me/svg/y%20%5Cin%20dom(f)" alt="y \in dom(f)" /> sao cho <img src="https://i.upmath.me/svg/f(y)%20%3C%20f(x%5E*)" alt="f(y) &lt; f(x^*)" />.  
Đặt <img src="https://i.upmath.me/svg/y'%3A%3D%20%5Clambda%20x%5E*%20%2B(1-%5Clambda)y" alt="y':= \lambda x^* +(1-\lambda)y" /> với <img src="https://i.upmath.me/svg/%5Clambda%20%5Cin%20(0%2C%201)" alt="\lambda \in (0, 1)" />.  
Từ tính chất của hàm lồi ta có:  
<img src="https://i.upmath.me/svg/f(y')%3Df(%5Clambda%20x%5E*%20%2B(1-%5Clambda)y)%20%5Cle%20%5Clambda%20f(x%5E*)%2B%20(1-%5Clambda)f(y)%20%3C%20%5Clambda%20f(x%5E*)%2B%20(1-%5Clambda)f(x%5E*)%3D%20f(x%5E*)" alt="f(y')=f(\lambda x^* +(1-\lambda)y) \le \lambda f(x^*)+ (1-\lambda)f(y) &lt; \lambda f(x^*)+ (1-\lambda)f(x^*)= f(x^*)" />  
Do đó <img src="https://i.upmath.me/svg/f(y')%20%3C%20f(x%5E*)" alt="f(y') &lt; f(x^*)" />. Chọn <img src="https://i.upmath.me/svg/%5Clambda" alt="\lambda" /> càng gần 1 thì <img src="https://i.upmath.me/svg/%5C%7Cy'-x%5E*%5C%7C%20%3C%20%5Cepsilon" alt="\|y'-x^*\| &lt; \epsilon" />. Điều này mẫu thuẫn với <img src="https://i.upmath.me/svg/x%5E*" alt="x^*" /> là local minimum.  
Từ đây suy ra điều phải chứng minh.  
Để ý thấy hàm Mean Square Error (MSE) trong Linear Regression hay Cross Entropy **trong Logistic Regression** đều là các hàm lồi. Các bạn có thể tự kiểm chứng điều này hoặc xem tại [Binary cross-entropy and logistic regression](https://towardsdatascience.com/binary-cross-entropy-and-logistic-regression-bf7098e75559?fbclid=IwAR1kSrG7pKJQvmge-M14CUkhjsZ0nlFA1Tw_4tBDWnBkBP8_fblXLrylk3s). Khi đó dùng thuật toán tối ưu Gradient Descent thì sẽ tìm được điểm tối ưu toàn cục.  
Thường các hàm lỗi trong Deep Neural Network là các hàm non-convex và đây là vấn đề của Non-convex Optimization.  
- Định lý: Cho <img src="https://i.upmath.me/svg/f%3A%20dom(f)%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: dom(f) \rightarrow \mathbb{R}" /> là hàm lồi chặt (strictly convex). Khi đó <img src="https://i.upmath.me/svg/f" alt="f" /> có điểm global minimum duy nhất.
### II. Phân tích tính hội tụ của Gradient Descent
Cho hàm <img src="https://i.upmath.me/svg/f%3A%20%5Cmathbb%7BR%7D%5Ed%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: \mathbb{R}^d \rightarrow \mathbb{R}" /> có đạo hàm và là hàm lồi. Giả sử rằng <img src="https://i.upmath.me/svg/f" alt="f" /> có điểm global minimum <img src="https://i.upmath.me/svg/x%5E*" alt="x^*" /> và mục tiêu là đi tìm (hoặc xấp xỉ) <img src="https://i.upmath.me/svg/x%5E*" alt="x^*" />, nghĩa là với <img src="https://i.upmath.me/svg/%20%5Cepsilon%20%3E0" alt=" \epsilon &gt;0" /> cho trước, ta cần tìm <img src="https://i.upmath.me/svg/x%20%5Cin%20%5Cmathbb%7BR%7D%5Ed" alt="x \in \mathbb{R}^d" /> sao cho <img src="https://i.upmath.me/svg/f(x)-f(x%5E*)%20%3C%20%5Cepsilon" alt="f(x)-f(x^*) &lt; \epsilon" />.  
Nhắc lại công thức cập nhật của Gradient Descent  <img src="https://i.upmath.me/svg/%5Ctheta%20%3D%20%5Ctheta%20-%20%5Ceta%20%5Cnabla_%7B%5Ctheta%7D%20f(%5Ctheta)" alt="\theta = \theta - \eta \nabla_{\theta} f(\theta)" /> hay viết lại <img src="https://i.upmath.me/svg/x_%7Bt%2B1%7D%3A%3Dx_t%20-%20%5Ceta%20%5Cnabla%20f(x_t)" alt="x_{t+1}:=x_t - \eta \nabla f(x_t)" />. Chúng ta muốn tìm số vòng lặp <img src="https://i.upmath.me/svg/t" alt="t" /> (số lần cập nhật) nhỏ nhất mà thỏa mãn <img src="https://i.upmath.me/svg/f(x_t)-f(x%5E*)%20%3C%20%5Cepsilon" alt="f(x_t)-f(x^*) &lt; \epsilon" />.  
#### Phần đầu tiên chúng ta tiếp cận với hàm Lipschitz Convex Function  
- **Định nghĩa:** (L-Lipschitz). Một hàm <img src="https://i.upmath.me/svg/f%3A%20%5COmega%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt="f: \Omega \rightarrow \mathbb{R}" /> được gọi là L-Lipschitz nếu với mọi <img src="https://i.upmath.me/svg/x%2C%20y%20%5Cin%20%5COmega" alt="x, y \in \Omega" /> ta có <img src="https://i.upmath.me/svg/%5C%7Cf(x)-f(y)%5C%7C%20%5Cle%20L%5C%7Cx-y%5C%7C" alt="\|f(x)-f(y)\| \le L\|x-y\|" />.  
- **Định lý:** Nếu hàm <img src="https://i.upmath.me/svg/f" alt="f" /> là L-Lipschitz, có đạo hàm và là hàm lồi, thì <img src="https://i.upmath.me/svg/%5C%7C%5Cnabla%20f(x)%5C%7C%20%5Cle%20L%20" alt="\|\nabla f(x)\| \le L " /> với mọi <img src="https://i.upmath.me/svg/x%20%5Cin%20%5COmega" alt="x \in \Omega" />.  
- **Định lý:** Cho <img src="https://i.upmath.me/svg/%20f%3A%20%5Cmathbb%7BR%7D%5Ed%20%5Crightarrow%20%5Cmathbb%7BR%7D" alt=" f: \mathbb{R}^d \rightarrow \mathbb{R}" /> là hàm lồi, có đạo hàm và L-Lipschitz trên toàn miền xác định của <img src="https://i.upmath.me/svg/f" alt="f" />, <img src="https://i.upmath.me/svg/f" alt="f" /> có global minimum <img src="https://i.upmath.me/svg/x%5E*" alt="x^*" />. Hơn nữa, giả sử <img src="https://i.upmath.me/svg/%5C%7Cx_0%20-%20x%5E*%5C%7C%20%5Cle%20R" alt="\|x_0 - x^*\| \le R" /> (<img src="https://i.upmath.me/svg/x_0" alt="x_0" /> là điểm khởi tạo ban đầu) và <img src="https://i.upmath.me/svg/%5C%7C%5Cnabla%20f(x)%5C%7C%20%5Cle%20L" alt="\|\nabla f(x)\| \le L" /> với mọi <img src="https://i.upmath.me/svg/x" alt="x" />. Chọn learning rate <img src="https://i.upmath.me/svg/%5Ceta%20%3D%5Cdfrac%7BR%7D%7BL%5Csqrt%7BT%7D%7D" alt="\eta =\dfrac{R}{L\sqrt{T}}" />. Gọi <img src="https://i.upmath.me/svg/x_1%2C%20x_2%2C..%2C%20x_%7BT-1%7D" alt="x_1, x_2,.., x_{T-1}" /> là dãy các giá trị <img src="https://i.upmath.me/svg/x" alt="x" /> được tính bằng công thức cập nhật của Gradient Descent với <img src="https://i.upmath.me/svg/%5Ceta%20%3D%5Cdfrac%7BR%7D%7BL%5Csqrt%7BT%7D%7D" alt="\eta =\dfrac{R}{L\sqrt{T}}" />.  
Khi đó ta có  <img src="https://i.upmath.me/svg/f%5Cbigg(%5Cdfrac%7B1%7D%7BT%7D%5Cdisplaystyle%20%5Csum_%7Bt%3D0%7D%5E%7BT-1%7Dx_t%5Cbigg)%20-%20f(x%5E*)%20%5Cle%20%5Cdfrac%7BRL%7D%7B%5Csqrt%7BT%7D%7D" alt="f\bigg(\dfrac{1}{T}\displaystyle \sum_{t=0}^{T-1}x_t\bigg) - f(x^*) \le \dfrac{RL}{\sqrt{T}}" />.  
**Chứng minh:**  
Đặt <img src="https://i.upmath.me/svg/g_t%20%3D%5Cnabla%20f(x_t)" alt="g_t =\nabla f(x_t)" />. Khi đó <img src="https://i.upmath.me/svg/g_t%20%3D%20%5Cdfrac%7Bx_t-x_%7Bt%2B1%7D%7D%7B%5Ceta%7D" alt="g_t = \dfrac{x_t-x_{t+1}}{\eta}" />  
<img src="https://i.upmath.me/svg/g_t%5ET(x_t%20-%20x%5E*)%3D%20%5Cdfrac%7B1%7D%7B%5Ceta%7D(x_t%20-x_%7Bt%2B1%7D)%5ET(x_t%20-%20x%5E*)" alt="g_t^T(x_t - x^*)= \dfrac{1}{\eta}(x_t -x_{t+1})^T(x_t - x^*)" />.  
Áp dụng công thức <img src="https://i.upmath.me/svg/2v%5ETw%20%3D%20%5C%7Cv%5C%7C%5E2%20%2B%5C%7Cw%5C%7C%5E2-%20%5C%7Cv-w%5C%7C%5E2" alt="2v^Tw = \|v\|^2 +\|w\|^2- \|v-w\|^2" /> với <img src="https://i.upmath.me/svg/v%3D%20g_t%2C%20w%3Dx_t%20-%20x%5E*" alt="v= g_t, w=x_t - x^*" />.  
Ta có <img src="https://i.upmath.me/svg/%20g_t%5ET(x_t%20-%20x%5E*)%3D%5Cdfrac%7B1%7D%7B2%5Ceta%7D(%5C%7Cx_t%20-%20x_%7Bt%2B1%7D%5C%7C%5E2%20%2B%5C%7Cx_t%20-%20x%5E*%5C%7C%5E2%20-%5C%7Cx_%7Bt%2B1%7D-x%20%5E*%5C%7C%5E2)%3D%20%20%20%20" alt=" g_t^T(x_t - x^*)=\dfrac{1}{2\eta}(\|x_t - x_{t+1}\|^2 +\|x_t - x^*\|^2 -\|x_{t+1}-x ^*\|^2)=    " />
<img src="https://i.upmath.me/svg/%20%5Cdfrac%7B1%7D%7B2%5Ceta%7D(%5Ceta%5E2%5C%7Cg_t%5C%7C%5E2%20%2B%5C%7Cx_t%20-%20x%5E*%5C%7C%5E2%20-%5C%7Cx_%7Bt%2B1%7D-x%20%5E*%5C%7C%5E2)%3D%20%20" alt=" \dfrac{1}{2\eta}(\eta^2\|g_t\|^2 +\|x_t - x^*\|^2 -\|x_{t+1}-x ^*\|^2)=  " />
<img src="https://i.upmath.me/svg/%5Cdfrac%7B%5Ceta%7D%7B2%7D%5C%7Cg_t%5C%7C%5E2%20%2B%5Cdfrac%7B1%7D%7B2%5Ceta%7D(%5C%7Cx_t%20-%20x%5E*%5C%7C%5E2%20-%5C%7Cx_%7Bt%2B1%7D-x%20%5E*%5C%7C%5E2)" alt="\dfrac{\eta}{2}\|g_t\|^2 +\dfrac{1}{2\eta}(\|x_t - x^*\|^2 -\|x_{t+1}-x ^*\|^2)" />.  
Lấy tổng trên <img src="https://i.upmath.me/svg/T" alt="T" /> lần cập nhật đầu tiên:  
<img src="https://i.upmath.me/svg/%5Cdisplaystyle%20%5Csum_%7Bt%3D0%7D%5E%7BT-1%7Dg_t%5ET(x_t%20-%20x%5E*)%3D%20%5Cdfrac%7B%5Ceta%7D%7B2%7D%5Csum_%7Bt%3D0%7D%5E%7BT-1%7D%5C%7Cg_t%5C%7C%5E2%20%2B%20%5Cdfrac%7B1%7D%7B2%5Ceta%7D(%5C%7Cx_0%20-%20x%5E*%5C%7C%5E2%20-%5C%7Cx_%7BT%7D-x%20%5E*%5C%7C%5E2)" alt="\displaystyle \sum_{t=0}^{T-1}g_t^T(x_t - x^*)= \dfrac{\eta}{2}\sum_{t=0}^{T-1}\|g_t\|^2 + \dfrac{1}{2\eta}(\|x_0 - x^*\|^2 -\|x_{T}-x ^*\|^2)" />  
Sử dụng tính chất first-order characterization of convexity ta có:  
<img src="https://i.upmath.me/svg/f(y)%20%5Cge%20f(x)%2B%20%5Cnabla%20f(x)%5ET(y-x)" alt="f(y) \ge f(x)+ \nabla f(x)^T(y-x)" /> với mọi <img src="https://i.upmath.me/svg/x%2C%20y" alt="x, y" />.  
Chọn <img src="https://i.upmath.me/svg/x%3Dx_t%2C%20y%3Dx%5E*" alt="x=x_t, y=x^*" /> ta được <img src="https://i.upmath.me/svg/f(x_t)%20-f%20(x%5E*)%20%5Cle%20g_t%5ET(x_t%20-%20x%5E*)" alt="f(x_t) -f (x^*) \le g_t^T(x_t - x^*)" />.  
Khi đó <img src="https://i.upmath.me/svg/%5Cdisplaystyle%20%5Csum_%7Bt%3D0%7D%5E%7BT-1%7D(f(x_t)-f(x%5E*))%20%5Cle%20%5Cdfrac%7B%5Ceta%7D%7B2%7D%5Csum_%7Bt%3D0%7D%5E%7BT-1%7D%5C%7Cg_t%5C%7C%5E2%20%2B%5Cdfrac%7B1%7D%7B2%5Ceta%7D%5C%7Cx_0%20-%20x%5E*%5C%7C%5E2" alt="\displaystyle \sum_{t=0}^{T-1}(f(x_t)-f(x^*)) \le \dfrac{\eta}{2}\sum_{t=0}^{T-1}\|g_t\|^2 +\dfrac{1}{2\eta}\|x_0 - x^*\|^2" />.  
Biểu thức này gọi là chặn trên của trung bình lỗi <img src="https://i.upmath.me/svg/f(x_t)-f(x%5E*)" alt="f(x_t)-f(x^*)" /> trên tất cả các lần cập nhật.  
Do <img src="https://i.upmath.me/svg/%5C%7Cx_0%20-%20x%5E*%5C%7C%5Cle%20R" alt="\|x_0 - x^*\|\le R" /> và <img src="https://i.upmath.me/svg/%5C%7Cg_t%5C%7C%20%5Cle%20L" alt="\|g_t\| \le L" /> nên   
<img src="https://i.upmath.me/svg/%5Cdisplaystyle%20%5Csum_%7Bt%3D0%7D%5E%7BT-1%7D(f(x_t)-f(x%5E*))%20%5Cle%20%5Cdfrac%7B%5Ceta%7D%7B2%7D%5Csum_%7Bt%3D0%7D%5E%7BT-1%7D%5C%7Cg_t%5C%7C%5E2%20%2B%5Cdfrac%7B1%7D%7B2%5Ceta%7D%5C%7Cx_0%20-%20x%5E*%5C%7C%5E2" alt="\displaystyle \sum_{t=0}^{T-1}(f(x_t)-f(x^*)) \le \dfrac{\eta}{2}\sum_{t=0}^{T-1}\|g_t\|^2 +\dfrac{1}{2\eta}\|x_0 - x^*\|^2" />  
<img src="https://i.upmath.me/svg/%5Cle%20%5Cdfrac%7B%5Ceta%7D%7B2%7DL%5E2T%2B%5Cdfrac%7B1%7D%7B2%5Ceta%7DR%5E2" alt="\le \dfrac{\eta}{2}L^2T+\dfrac{1}{2\eta}R^2" />.  
Xét hàm <img src="https://i.upmath.me/svg/h(%5Ceta)%3D%20%5Cdfrac%7B%5Ceta%7D%7B2%7DL%5E2T%2B%5Cdfrac%7BR%5E2%7D%7B2%5Ceta%7D" alt="h(\eta)= \dfrac{\eta}{2}L^2T+\dfrac{R^2}{2\eta}" />.  
Ta chọn <img src="https://i.upmath.me/svg/%5Ceta" alt="\eta" /> sao cho <img src="https://i.upmath.me/svg/h" alt="h" /> đạt giá trị nhỏ nhất.  
Giải phương trình đạo hàm <img src="https://i.upmath.me/svg/q'(%5Ceta)%3D0" alt="q'(\eta)=0" /> được <img src="https://i.upmath.me/svg/%5Ceta%3D%5Cdfrac%7BR%7D%7BL%5Csqrt%7BT%7D%7D" alt="\eta=\dfrac{R}{L\sqrt{T}}" /> và <img src="https://i.upmath.me/svg/h(%5Cdfrac%7BR%7D%7BL%5Csqrt%7BT%7D%7D)%3D%20%5Cdfrac%7BRL%7D%7B%5Csqrt%7BT%7D%7D" alt="h(\dfrac{R}{L\sqrt{T}})= \dfrac{RL}{\sqrt{T}}" />.  
Từ đây dễ thấy với <img src="https://i.upmath.me/svg/T%20%5Cge%20%5Cdfrac%7BR%5E2B%5E2%7D%7B%5Cepsilon%5E2%7D" alt="T \ge \dfrac{R^2B^2}{\epsilon^2}" /> thì average error <img src="https://i.upmath.me/svg/%5Cle" alt="\le" /> <img src="https://i.upmath.me/svg/%5Cdfrac%7BLB%7D%7B%5Csqrt%7BT%7D%7D%20%5Cle%20%5Cepsilon" alt="\dfrac{LB}{\sqrt{T}} \le \epsilon" />.  
Và do <img src="https://i.upmath.me/svg/f" alt="f" /> là hàm lồi nên <img src="https://i.upmath.me/svg/f%5Cbigg(%5Cdfrac%7B1%7D%7BT%7D%5Cdisplaystyle%20%5Csum_%7Bt%3D0%7D%5E%7BT-1%7Dx_t%5Cbigg)%20-%20f(x%5E*)%20%5Cle%20%5Cdfrac%7BRL%7D%7B%5Csqrt%7BT%7D%7D%20%5Cblacksquare" alt="f\bigg(\dfrac{1}{T}\displaystyle \sum_{t=0}^{T-1}x_t\bigg) - f(x^*) \le \dfrac{RL}{\sqrt{T}} \blacksquare" />.  
Suy ra độ phức tạp của Gradient Descent đối với hàm Lipschitz Convex Function là <img src="https://i.upmath.me/svg/O%5Cbigg(%5Cdfrac%7B1%7D%7B%5Cepsilon%5E2%7D%5Cbigg)" alt="O\bigg(\dfrac{1}{\epsilon^2}\bigg)" />.  
Ở bài sau mình sẽ nói về tốc độ hội tụ của Gradient Descent trên các hàm **Smooth Convex Function** và **Strongly Convex Function**.  
### III. Tài liệu tham khảo
[1]. https://machinelearningcoban.com/  
[2]. https://github.com/epfml/OptML_course  
[3]. https://ee227c.github.io/  
[4]. Stephen Boyd and Lieven Vandenberghe.  
Convex Optimization.  
Cambridge University Press, New York, NY, USA, 2004.  
https://web.stanford.edu/~boyd/cvxbook/.  
[5]. https://easyai.tech/en/ai-definition/gradient-descent/  
[6]. https://towardsdatascience.com/binary-cross-entropy-and-logistic-regression-bf7098e75559?fbclid=IwAR1kSrG7pKJQvmge-M14CUkhjsZ0nlFA1Tw_4tBDWnBkBP8_fblXLrylk3s


























