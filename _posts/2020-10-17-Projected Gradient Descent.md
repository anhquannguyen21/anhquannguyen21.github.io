---
layout: post
title: Projected Gradient Descent
tags: [Convex Optimization]
---





<p>Ở các bài trước, chúng ta sử dụng gradient descent với miền của <img src="https://i.upmath.me/svg/x" alt="x" /> là <img src="https://i.upmath.me/svg/x%20%5Cin%20%5Cmathbb%7BR%7D%5Ed" alt="x \in \mathbb{R}^d" />. Khi miền <img src="https://i.upmath.me/svg/X%20%5Cne%20%5Cmathbb%7BR%7D%5Ed" alt="X \ne \mathbb{R}^d" /> thì điểm cập nhật mới <img src="https://i.upmath.me/svg/x_%7Bt%2B1%7D" alt="x_{t+1}" /> có thể không thuộc <img src="https://i.upmath.me/svg/X" alt="X" />. Do đó sau mỗi bước cập nhật ta phải chiếu nó xuống <img src="https://i.upmath.me/svg/X" alt="X" /> để thu được điểm thuộc <img src="https://i.upmath.me/svg/X" alt="X" />. Đây là một bài toán tối ưu có ràng buộc (constrained optimzation).<br>
<a href="https://postimg.cc/d73yMnDp"><img src="https://i.postimg.cc/rpCCrPTV/pgd.png" alt="pgd.png"></a></p>
<p>Trong bài này mình sẽ trình bày về <strong>Projected Gradient Descent</strong>.
<a href="https://postimg.cc/HjcFWCfs"><img src="https://i.postimg.cc/66YK0WcR/pgd.png" alt="pgd.png"></a></p>
<p>Sau mỗi lần cập nhật, thuật toán sẽ chiếu điểm xuống miền <img src="https://i.upmath.me/svg/X" alt="X" />.<br>
<a href="https://postimg.cc/jD7zCMRf"><img src="https://i.postimg.cc/HkS3ZNK6/pgd.png" alt="pgd.png"></a></p>
<p>Ta có công thức cập nhật của projected gradient descent như sau:<br>
<img src="https://i.upmath.me/svg/y_%7Bt%2B1%7D%3Dx_t%20-%20%5Ceta%5Cnabla%20f(x_t)" alt="y_{t+1}=x_t - \eta\nabla f(x_t)" /><br>
<img src="https://i.upmath.me/svg/x_%7Bt%2B1%7D%3D%5CPi_X(y_%7Bt%2B1%7D)%3D%5Carg%20%5Cmin_%7Bx%20%5Cin%20X%7D%5C%7C%20x-y_%7Bt%2B1%7D%5C%7C%5E2." alt="x_{t+1}=\Pi_X(y_{t+1})=\arg \min_{x \in X}\| x-y_{t+1}\|^2." /><br>
Trước hết ta có một số tính chất sau:<br>
Cho <img src="https://i.upmath.me/svg/X%20%5Csubseteq%20%5Cmathbb%7BR%7D%5Ed" alt="X \subseteq \mathbb{R}^d" /> là tập đóng và lồi. Cho <img src="https://i.upmath.me/svg/x%20%5Cin%20X%2C%20y%20%5Cin%20%5Cmathbb%7BR%7D%5Ed" alt="x \in X, y \in \mathbb{R}^d" />. Khi đó<br>
<strong>i))</strong> <img src="https://i.upmath.me/svg/(x-%5CPi_X(y))%5E%7BT%7D(y-%20%5CPi_X(y))%20%5Cle%200" alt="(x-\Pi_X(y))^{T}(y- \Pi_X(y)) \le 0" /><br>
<strong>ii))</strong> <img src="https://i.upmath.me/svg/%5C%7Cx-%5CPi_X(y)%5C%7C%5E2%2B%5C%7Cy-%20%5CPi_X(y)%5C%7C%5E2%20%5Cle%20%5C%7Cx-y%5C%7C%5E2" alt="\|x-\Pi_X(y)\|^2+\|y- \Pi_X(y)\|^2 \le \|x-y\|^2" />
<a href="https://postimg.cc/RNgBNzM3"><img src="https://i.postimg.cc/pLHXc24B/pgd.png" alt="pgd.png"></a><br>
<strong>Chứng minh</strong><br>
Ta có bổ đề quan trọng sau:<br>
<strong>Bổ đề</strong>: Cho hàm <img src="https://i.upmath.me/svg/f%3A%20dom(f)%20%5Crightarrow%20%5Cmathbb%7BR%7D%20" alt="f: dom(f) \rightarrow \mathbb{R} " /> là hàm lồi và khả vi trên toàn miền <img src="https://i.upmath.me/svg/dom(f)%20%5Csubseteq%20%5Cmathbb%7BR%7D%5Ed" alt="dom(f) \subseteq \mathbb{R}^d" /> và cho <img src="https://i.upmath.me/svg/X%20%5Csubseteq%20dom(f)" alt="X \subseteq dom(f)" /> là tập lồi. Điểm <img src="https://i.upmath.me/svg/x%5E*" alt="x^*" /> được gọi là <strong>minimizer</strong> của <img src="https://i.upmath.me/svg/f" alt="f" /> trên toàn miền <img src="https://i.upmath.me/svg/X" alt="X" /> nếu và chỉ nếu <img src="https://i.upmath.me/svg/%5Cnabla%20f(x%5E*)(x-x%5E*)%20%5Cge%200" alt="\nabla f(x^*)(x-x^*) \ge 0" /> với mọi <img src="https://i.upmath.me/svg/x%20%5Cin%20X" alt="x \in X" />.<br>
<strong>i)</strong> Dễ thấy <img src="https://i.upmath.me/svg/%5CPi_X(y)" alt="\Pi_X(y)" /> là minimizer của hàm lồi <img src="https://i.upmath.me/svg/d_y(x)%3D%5C%7Cx-y%5C%7C%5E2" alt="d_y(x)=\|x-y\|^2" /> trên miền <img src="https://i.upmath.me/svg/X" alt="X" />. Sử dụng bổ đề trên ta có:<br>
<img src="https://i.upmath.me/svg/%5Cnabla%20d_y(%5CPi_X(y))%5ET(x-%5CPi_X(y))%20%5Cge%200" alt="\nabla d_y(\Pi_X(y))^T(x-\Pi_X(y)) \ge 0" /><br>
<img src="https://i.upmath.me/svg/%5Ciff%202(%5CPi_X(y)-y)%5ET(x-%5CPi_X(y))%20%5Cge%200" alt="\iff 2(\Pi_X(y)-y)^T(x-\Pi_X(y)) \ge 0" /><br>
<img src="https://i.upmath.me/svg/%5Ciff%20(x-%5CPi_X(y))%5ET(%5CPi_X(y)-y)%20%5Cle%200%20" alt="\iff (x-\Pi_X(y))^T(\Pi_X(y)-y) \le 0 " /> <strong>(đpcm)</strong></p>
<p><strong>ii)</strong> Đặt <img src="https://i.upmath.me/svg/v%3Dx-%5CPi_X(y)%2C%20w%3D%5CPi_X(y)-y" alt="v=x-\Pi_X(y), w=\Pi_X(y)-y" />. Sử dụng <strong>i)</strong> ta có<br>
<img src="https://i.upmath.me/svg/0%20%5Cge%202v%5ETw%3D%20%5C%7Cv%5C%7C%5E2%20%2B%20%5C%7Cw%5C%7C%5E2%20%2B%5C%7Cv-w%5C%7C%5E2%3D%20%7Cx-%5CPi_X(y)%5C%7C%5E2%2B%5C%7Cy-%20%5CPi_X(y)%5C%7C%5E2-%20%5C%7Cx-y%5C%7C%5E2" alt="0 \ge 2v^Tw= \|v\|^2 + \|w\|^2 +\|v-w\|^2= |x-\Pi_X(y)\|^2+\|y- \Pi_X(y)\|^2- \|x-y\|^2" />.<br>
Suy ra <img src="https://i.upmath.me/svg/%5C%7Cx-%5CPi_X(y)%5C%7C%5E2%2B%5C%7Cy-%20%5CPi_X(y)%5C%7C%5E2%20%5Cle%20%5C%7Cx-y%5C%7C%5E2" alt="\|x-\Pi_X(y)\|^2+\|y- \Pi_X(y)\|^2 \le \|x-y\|^2" /> <strong>(đpcm)</strong><br>
Việc phân tích tính hội tụ của Projected Gradient Descent trên các dạng hàm Lipschitz Convex Function, Smooth Convex Function, Smooth and Strongly Convex Function hoàn toàn tương tự ở các bài trước về Gradient Descent.</p>
<h4>Tài liệu tham khảo</h4>
<ol>
<li><a href="https://github.com/epfml/OptML_course">https://github.com/epfml/OptML_course</a></li>
<li><a href="https://ee227c.github.io/">https://ee227c.github.io/</a></li>
<li>Stephen Boyd and Lieven Vandenberghe.<br>
Convex Optimization.<br>
Cambridge University Press, New York, NY, USA, 2004.</li>
</ol>