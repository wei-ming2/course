### 1. Motivation
- **Classification**: output variable $y$ can take on only a handful of small possible values

- **Eg**: binary classification (yes 1 / no 0 output)
	- Is email spam? (Yes / No)
	- Is transaction fraudulent? (Yes / No)
	- Is tumour malignant? (Yes / No)

- **Linear Regression for Classification**:  <br> ![[Screenshot 2026-09-04 at 10.05.26 AM.png|500]]
	- Tumour size can vary greatly, linear line will not be able to properly fit / classify all the results well

#### 1.1. Logistic Regression
- **Logistic regression**: creates a "S" like curve (sigmoid function) <br> ![[Screenshot 2026-09-04 at 10.18.15 AM.png|300]]
	- Sigmoid function: ranges from 0 (as $x\to -\infty$) to 1 (as $x \to \infty$)
	- Anything above threshold (0.7) is classified to be 1, otherwise 0

- **Formula**: $$\begin{align*}f_{\vec{w},b}(\vec{x})=\vec{w}\cdot\vec{x}+b&=z \\
  g(z)&=\frac{1}{1+e^{-z}}\end{align*}$$
	- Fundamentally, **logistic regression** is a **generalised linear regression** except it is remapped over to a new scale (in most cases 0 to 1)

#### 1.2. Decision Boundary
- **Decision boundary**: separation line, curve, surface that a classification model create to separate groups or classes of data <br> ![[Screenshot 2026-09-04 at 11.05.18 AM.png|300]] $$z=x_1+x_2-3=0$$
	- $'\text{X}'$ denotes $1$, $'\bigcirc'$ denotes $0$
	- TLDR: the purple line, anything right / above is $1$, and anything left / below is $0$

- **Non-linear decision boundaries**: <br>  ![[Screenshot 2026-09-08 at 3.30.31 PM.png|300]]$$z=x_1^2+x_2^2-1=0$$

### 2. Cost Function
- **Cost function**: measures how well a specific set of parameters fits the training data, thereby giving us a way to better choose better parameter

#### 2.1. Cost Function for Logistic Regression
- **Squared error cost function**: for linear, not ideal for logistic regression ![[wk1 - Introduction to Machine Learning#2.2. Cost Function]]
	- **Applied to logistic regression**: non-convex cost function <br> ![[Screenshot 2026-09-10 at 9.03.11 AM.png|300]]
		- Lots of local minima the function could get stuck in

- **Updated cost function**: move $\frac{1}{2}$ from outside $(\frac{1}{2m})$ to inside  $$J(\vec{w},b)=\frac{1}{m}\sum_{i=1}^m\frac{1}{2}(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})^2$$

- **Logistic regression loss function**: above to reach a global minimum <br> ![[Screenshot 2026-09-10 at 9.47.54 AM.png|300]] $$L(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})=
  \begin{cases}-\log(f_{\vec{w},b}(\vec{x}^{(i)})) \hspace{1.3cm}\text{if } y^{(i)}=1 \\
  -\log(1-f_{\vec{w},b}(\vec{x}^{(i)})) \hspace{0.5cm}\text{if } y^{(i)}=0\end{cases}$$
	- $y^{(i)}$: prediction
	- TLDR: prediction of logistic regression $0-1$, the new loss function heavily penalises confidently wrong prediction
		- To take note of: mislabeled data, measurement error, extreme outlier

#### 2.2. Simplified Loss Function
- Convert the 2 $\log$ case formula: $$L(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})=-y^{(i)}\log(f_{\vec{w},b}(\vec{x}^{(i)}))- (1-y^{(i)})\log(1-f_{\vec{w},b}(\vec{x}^{(i)}))$$
	- Note: $y$ can only take on values $0,1$, so either one of the value will always be $0$ $(0-x | -x-0)$

- **Simplified cost function**: $$\begin{align*}J(\vec{w},b)
  &=\frac{1}{m}\sum_{i=1}^m[L(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})] \\ 
  &=-\frac{1}{m}\sum_{i=1}^m[y^{(i)}\log(f_{\vec{w},b}(\vec{x}^{(i)}))+(1-y^{(i)})\log(1-f_{\vec{w},b}(\vec{x}^{(i)}))]\end{align*}$$

#### 2.3. Gradient Descent Implementation
- **Gradient Descent Derivatives**: $$\begin{align*}
  \frac{\partial}{\partial w_j}J(\vec{w},b)&=\frac{1}{m}\sum_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)} \\
  \frac{\partial}{\partial b}J(\vec{w},b)&=\frac{1}{m}\sum_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})
  \end{align*}$$

- Even though the gradient descent algorithm is the same, because $f$ is different, the overall loss functions are different 

| Linear Regression                              | Logistic Regression                                               |
| ---------------------------------------------- | ----------------------------------------------------------------- |
| $f_{\vec{w},b}(\vec{x})=\vec{w}\cdot\vec{x}+b$ | $f_{\vec{w},b}(\vec{x})=\frac{1}{1+e^{(-\vec{w}\cdot\vec{x}+b)}}$ |
