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

### 3. The Problem of Overfitting

| Type                               | Graph                                              | Remark                                                                                                         |
| ---------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Under-fitting**                  | ![[Screenshot 2026-09-11 at 10.41.32 AM.png\|300]] | Model does not fit the training data well (high bias)<br><br>(Price eventually flattens out as size increases) |
| **Generalisation**<br>(just right) | ![[Screenshot 2026-09-11 at 10.47.06 AM.png\|300]] | Fits the training set pretty well (and makes sense)                                                            |
| **Overfitting**                    | ![[Screenshot 2026-09-11 at 10.49.47 AM.png\|300]] | Fits the training set extremely well, cost may equal 0 (high variance)<br><br>(But do not make sense)          |
- **Linear regression** & **Classification**

| Under-fitting                                 | Generalisation                                | Overfitting                                   |
| --------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| ![[Screenshot 2026-09-11 at 10.56.13 AM.png]] | ![[Screenshot 2026-09-11 at 10.56.53 AM.png]] | ![[Screenshot 2026-09-11 at 10.57.26 AM.png]] |
| Does not capture the general trend            | Seems about right                             | Overly complicated with no trend              |


#### 3.1. Addressing Overfitting
1. **Collect more training examples**: such that the model has less room for wiggly pattern <br> ![[Screenshot 2026-09-11 at 11.01.54 AM.png|300]]![[Screenshot 2026-09-11 at 11.02.12 AM.png|300]]

2. **Reduce the number of features**: reduce complexity of polynomial features
	- Especially when the training data is few

3. **Regularisation**: encouraging the learning algorithm to shrink the parameters without demanding parameter is set to 0 $$f(x)=28x-385x^2+39x^3-174x^4+100$$
	- Essentially setting some of the feature to 0, $-174x^4\to 0$: $w_j$ values end up being smaller $$f(x)=13x-0.23x^2+0.000014x^3-0.0001x^4+10$$
	- **TLDR**: keeping the features whilst preventing it from having an overly large effect, which causes overfitting

#### 3.2. Cost Function with Regularisation
- **Intuition**: attempt to simplify the polynomial by getting the higher power to approach 0, such that the effect of it is essentially none 

- **Cost function with regularisation**: $$J(\vec{w},b)=\frac{1}{2m}\sum_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2+\frac{\lambda}{2m}\sum_{j=1}^nw_j^2$$
	- $\lambda$: regularisation parameter, $>0$
	- **New cost function**: MSE (fits the data) + regularisation term (applies penalty to keep $w_j$ from growing large)

| Lambda                                  | Plot                                              | Idea                                                                                                                                                                                                |
| --------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\lambda=0$                             | ![[Screenshot 2026-09-14 at 9.38.01 AM.png\|300]] | Effect of regularisation term is 0.<br><br>Model minimises error at all cost (thus overfit)                                                                                                         |
| $\lambda=10^{10}$ <br>(extremely large) | ![[Screenshot 2026-09-14 at 9.39.43 AM.png\|300]] | Model attempts to reduce the *regularisation term*; all $w_j\to0$ (thus under-fit)<br>$$\begin{aligned}f_{\vec{w},b}(\vec{x})&=w_1x+w_2x^2+w_3x^3+w_4x^4+b\\&\approx 0+0+0+0+b \\&=b\end{aligned}$$ |

#### 3.3. Gradient Descent with Regularisation
- **Gradient Descent Recap**: ![[wk1 - Introduction to Machine Learning#3.1. Implementing Gradient Descent]]
- **Regularised linear regression**: very similar except the derivative with respective to $w_j$ has the *regularisation term* $$\frac{\partial}{\partial w_j}J(\vec{w},b)=\frac{1}{m}\sum_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)})-y^{9i)})x_j^{(i)}+\frac{\lambda}{m}w_j$$
	- no change to update of $b$ $(\frac{\partial}{\partial b})$

- **Regularised logistic regression**: same formula as linear regression, except $f_{\vec{w},b}$ is different