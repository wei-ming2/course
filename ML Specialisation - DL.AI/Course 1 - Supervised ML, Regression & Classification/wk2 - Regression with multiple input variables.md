### 1. Multiple Features
- **Previously**: given $x$, predict $y$ (given size of house, predict the price) $$f_{w,b}(x)=wx+b$$
- **Now**: given $x_1,x_2,...,x_n$, predict $y$ (given size, bedroom, floor, age, predict the price)
	- $x_j=j^{th}$ feature
	- $n$: no. features
	- $\vec{x}^{(i)}$: features of $i^{th}$ training example
		- $\vec{x}^{(2)}$: $[1416, 3, 2, 40]$ (all the features of the 2nd row)
		- $x_3^{(2)}$: 2 (selecting the 3rd feature of the 2nd row)

- **Overall**: multiple linear regression (feature) is therefore $$\begin{align*}f_{w,b}(x)&=w_1x_1+w_2x_2+...+w_nx_n+b \\
  &=\vec{w}\cdot\vec{x}+b\end{align*}$$
	- $\vec{w}=[w_1, w_2, ..., w_n]$: parameters of the model
	- $\vec{x}=[x_1, x_2, ..., x_n]$: features
	- (NOT multivariate regression)

### 2. Vectorisation
- Vectorisation shortens the code and makes it run more efficiently

```python
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10, 20, 30])
```
- **Difference between Implementation**:
```python
f = w[0] * x[0] + # sequential calculation
	w[1] * x[1] + 
	w[2] * x[2] + b
```

```python
f = np.dot(w, x) + b # vectorisation
```
- Vectorisation is able to run in parallel in the background, able to better utilise the hardware

- **Numpy methods introduced**:
	1. Vector creation
	2. Vector indexing
	3. Vector slicing
	4. Vector sum, mean, etc operations
	5. Vector element-wise operation
	6. Vector dot product
	7. Vector reshape

|                  | Previous Notation                                                                                                                                      | Vector Notation                                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Parameters       | $w_1, ..., w_n$<br>$b$                                                                                                                                 | $\vec{w}=[w1, ..., w_n]$<br>$b$ (same)                                                                                                         |
| Model            | $f_{\vec{w}, b}(\vec{x})=w_1x_1+...+w_nx_n+b$                                                                                                          | $f_{\vec{w},b}=\vec{w}\cdot\vec{x}+b$                                                                                                          |
| Cost function    | $J(w_1,...,w_n,b)$                                                                                                                                     | $J(\vec{w},b)$                                                                                                                                 |
| Gradient descent | repeat until convergence<br>$w_j=w_j-\alpha\frac{\partial}{\partial w_j}J(w_1,...,w_n,b))$<br>$b=b-\alpha\frac{\partial}{\partial b}J(w_1,...,w_n,b))$ | repeat until convergence<br>$w_j=w_j-\alpha\frac{\partial}{\partial w_j}J(\vec{w},b))$<br>$b=b-\alpha\frac{\partial}{\partial b}J(\vec{w},b))$ |

### 3. Gradient Descent for Multiple Linear Regression
- **Recap**: single linear regression ![[wk1 - Introduction to Machine Learning#3. Gradient Descent#3.4. Gradient Descent for Linear Regression]]
- $n$ features $(n\geq 2)$: $$w_n=w_n-\alpha\frac{1}{m}\sum_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_n^{(i)}$$
	- $j$ feature index (feature being updated): $j=1\to w_1, x_1$

- **Alternative to gradient descent**:
	- **Normal equation**
		- Only for linear regression
		- Solve for $w$, $b$ without iterations
	- **Disadvantages**
		- Do not generalise to other learning algorithms
		- Slow when feature is large ($> 10,000$)

#### 3.1. Feature Scaling
- Enable gradient descent to run much faster

- Essentially when $x_1$ and $x_2$ exists on different scale: 
	- $x_1$: size of house (sqft) $(100-2000+)$
	- $x_2$: bedroom $(1-5+)$

- The corresponding values of $w_1$ and $w_2$ will as such also exists on a different scale:
	- $w_1$: 0.1 (0.1 thousand per sqft)
	- $w_2$: 50 (50 thousand per bedroom)
	- 500x difference in the magnitude between gradient of $w_1$ and $w_2$

- **Feature scaling**: fixes the excessive difference in scale between parameters, back to a normal circular contour plot
![[Screenshot 2026-09-03 at 10.25.25 AM.png]]

- **Acceptable ranges**: typically aim for -1 to 1 $$\begin{align*}0&\leq x_1\leq3 \\
  -2&\leq x_2\leq 0.5\end{align*}$$
	- **Not acceptable**: $-100\leq x_3\leq 100$, $-0.001\leq x_4\leq 0.001$

#### 3.2. Normalisation
- **Mean normalisation**: centers data around 0, strictly bounded between -1 and 1 <br> ![[Screenshot 2026-09-03 at 10.32.11 AM.png|300]] $$x_1=\frac{x_1-\mu_1}{\text{max - min}}$$

- **Z-score normalisation**: mean 0 and SD becomes 1 (number of SD away) <br> ![[Screenshot 2026-09-03 at 10.41.34 AM.png|300]]$$x_1=\frac{x_1-\mu_1}{\sigma_1}$$

### 4. Gradient Descent for Convergence
- If we plot loss, $J(\vec{w},b)$ against *iterations*, the loss should go down after every iterations

- **Typically, the phases**: <br> ![[Screenshot 2026-09-03 at 10.52.36 AM.png|300]]
	1. Greatest descent
	2. Decreasing descent
	3. Curve flattens out (converged)
		- number of iterations may differ (30, 500, etc)

- **Automatic convergence test, $\epsilon$**: if $J(\vec{w},b)$ decreases by $\leq \epsilon$ in 1 iteration, declare convergences
	- Declare we found parameters $\vec{w},b$ close to global minimum
	- idea: at Phase 3, the curve seem to flatten out, but there may still be a decrease still of a very small magnitude.

#### 4.1. Learning Rate
- **Recap**: ![[wk1 - Introduction to Machine Learning#3.3. Learning Rate]]

- **Plot of loss against iterations can give you a good clue about the learning rate**: 
	- **General rule**: loss should decrease on every iteration
	- **Possible bugs**: 
		- **Loss strictly increasing**: adding rather than subtracting / $\alpha$ too large <br> ![[Screenshot 2026-09-03 at 11.04.01 AM.png|300]]$$w_1=w_1+\alpha d_1\to w_1=w_1-\alpha d_1$$
		- **Loss decreasing and increasing**: learning rate is too large and "jumping" <br> ![[Screenshot 2026-09-03 at 11.04.42 AM.png|300]]
		- **Debug**: use an extremely small $\alpha$ and loss should decrease slowly (if it still does not, there is likely error elsewhere)

- **Tip**: use many different $\alpha$ <br> ![[Screenshot 2026-09-03 at 11.08.47 AM.png|300]]$$..., 0.0001, 0.001, 0.01, 0.1, 1, ...$$
	- run a few iteration to pick the best $\alpha$ (decrease rapidly and consistently)
	- **Andrew's preference**: learning rate $\times 3$ $$..., 0.01, 0.03, 0.1, 0.3, 1, ...$$

### 5. Feature Engineering
- **Feature engineering**: using intuition to *design new features*, by *transforming*, or *combining original features*

- **Eg**: given that we know the frontage $x_1$ and depth $x_2$ <br> ![[Screenshot 2026-09-03 at 11.20.18 AM.png|300]] $$f_{\vec{w},b}(\vec{x})=w_1x_1+w_2x_2+b$$
	- However, from our intuition, what matters more is technically the area $$\text{area}=\text{frontage}\times\text{depth}\to x_3=x_1x_2$$
		- updated function: $$f_{\vec{w},b}(\vec{x})=w_1x_1+w_2x_2+w_3x_3+b$$

### 6. Polynomial Regression
- Involving a power of $x$ in prediction $(x^2,x^3, \sqrt{x},...)$

- **Eg**: <br> ![[Screenshot 2026-09-03 at 11.29.57 AM.png|300]]$$f_{\vec{w},b}(x)=w_1x+w_2x^2+b$$
	- limitation: $x^2$ eventually comes down (which does not make sense in the context)

- We can use $+w_3x^3$: $$f_{\vec{w},b}(x)=w_1x+w_2x^2+w_3x^3+b$$
	- but it is now extremely important for proper *feature scaling*, if $x$ is $(1-1000)$, cubic would mean it is $(1 - 1 \text{ billion})$ 
