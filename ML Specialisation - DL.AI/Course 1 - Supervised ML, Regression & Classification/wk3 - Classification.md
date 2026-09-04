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

