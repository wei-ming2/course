### 1. Supervised vs Unsupervised Machine learning
- **Machine Learning Algorithms**: 4 types
	1. Supervised learning
	2. Unsupervised learning
	3. Recommender systems
	4. Reinforcement learning

#### 1.1. Supervised Learning
- **Supervised learning**: learning from the right answers

- **Learn $x$, output $y$, input-to-output mappings**
	- Ie: $x$: email, $y$: spam/not-spam
	- spam filtering, speech recognition, machine translation, online advertising, self-driving car, visual inspection

- **Type**: 
	1. **Regression**: predict a number from infinitely many possible numbers (housing price from housing size) <br>![[Screenshot 2026-08-26 at 10.38.50 AM.png|300]]
	2. **Classification**: predict a small number of outcomes (using patients records etc to determine if there is breast cancer tumor or just a lump) <br>![[Screenshot 2026-08-26 at 10.37.08 AM.png|300]]

#### 1.2. Unsupervised Learning
- **Unsupervised learning**: find something interesting in unlabelled data

- **Type**:
	1. **Clustering**: place unlabelled data into different clusters (grouping related articles of Google news with related keyword: panda, zoo) <br>![[Screenshot 2026-08-26 at 10.48.43 AM.png|300]]
	2. **Anomaly detection**: find unusual data points (fraud detection in financial system)
	3. **Dimensionality reduction**: compress a large dataset while losing as little information as possible

### 2. Regression Model
#### 2.1. Linear Regression
- TLDR: fitting straight line to your data to represent the relationship

- **Notation**:
	- $x$: input variable / feature
	- $y$: output variable / target
	- $m$: no. of training examples
	- $(x,y)$: single training example
	- $(x^{(i)},y^{(i)})$: $i^{\text{th}}$ training example

- **Pipeline**:
	- training set (features, targets)
	- learning algorithm
	- $x\to f\to \hat{y}$ (feature -> model -> prediction)
		- representation of $f$: $$f_{w,b}(X)=f(X)=wx+b$$(univariate linear regression - 1 feature)

#### 2.2. Cost Function
- **Cost function**: tells us how well the model is doing

- **Error**: measuring how far off the prediction is from the target 
- **Squared error cost function**: $$J(w,b)=\frac{1}{2m}\sum_{i=1}^m(\hat{y}^{(i)}-y^{(i)})^2$$
	- where $\hat{y}^{(i)}=f_{w,b}(x^{(i)})$
	- goal: to reduce $J$

- **Function of $w$**:
	- Assume $\hat{y}^{(i)}=f_w(x^{(i)})=wx^{(i)}$, we can plot $J(w)$, the cost against the parameter $w$, which decides the gradient <br>![[Screenshot 2026-08-26 at 4.22.22 PM.png|500]]
		- the graph helps us estimate $w$ with the least cost, the best $w$

#### 2.3. Visualisation of Cost Function
- In 2D field, $y$ against $x$, the function of $w$ is shown to be a convex (bowl like graph), with the lowest loss existing at some point.

- When it is translated to $f_{w,b}(x)=wx^{(i)}+b$, there is a bowl in loss against $w$ and a bowl in loss against $b$: <br>![[Screenshot 2026-08-26 at 4.30.29 PM.png|300]]

- Alternatively, 3D plot can also be visualised in contour plot: <br>![[Screenshot 2026-08-28 at 9.31.28 AM.png|300]]
	- Distance between lines represent the height / value

- **Summary**: <br>![[Screenshot 2026-08-28 at 9.39.49 AM.png|500]]
	1. Make a $J$ (loss) against $w$ (gradient) plot, and draw a smooth line for the graph
	2. Make a $J$ (loss) against $b$ (shift) plot, and draw a smooth line for the graph
	3. Combine to $J$ against $w$ and $b$ 3D plot.
	4. Find the lowest point in the 3D space for $J$ (least loss) then select the corresponding $w$ and $b$ to get the best fit line

### 3. Gradient Descent
- **Idea**: given $J(w,b)$, we want to minimise $\underset{w,b}{\text{min}}\hspace{0.1cm}J(w,b)$
- **Outline**: 
	1. Start with some $w,b$ (typically set $w=0,b=0$)
	2. Keep changing $w,b$ to reduce $J(w,b)$ until we settle at or near a minimum (may have >1 minimum)

- **TLDR**: gradient descent is essentially starting at a random point (ie: hill), look for the greatest gradient descent (ie: place where you can descent the hill fastest), and move in that direction (ie: and take a baby step) repeatedly till we reached the minimum point (ie: till reach valley) <br>![[Screenshot 2026-08-31 at 8.39.15 AM.png|500]]
	- Different starting point may lead to different (local) minima 

#### 3.1. Implementing Gradient Descent
- **Gradient descent algorithm**: $$\begin{align*}w&=w-\alpha \frac{d}{dw}J(w,b) \\
  b&=b-\alpha\frac{d}{db}J(w,b)\end{align*}$$
	- $\alpha$: learning rate (typically between 0 and 1)
	- Simultaneously update $w$ (gradient) together with $b$ (the shift) $$\begin{align*}tmp\_w&=w-\alpha \frac{\partial}{\partial w}J(w,b) \\ 
	  tmp\_b&=b-\alpha\frac{\partial}{\partial b}J(w,b) \\ 
	  w&=tmp\_w \\
	  b&=tmp\_b\end{align*}$$
		- Use old $w,b$ to calculate $w$ and $b$ before updating them with the new values
	- Repeat the algorithm until convergence (till $w,b$ do not change much with every step taken)

#### 3.2. Gradient Descent Intuition
- **Negative of $\frac{\partial}{\partial w}J(w,b)$**: 
	- **Positive gradient**: (negative $\times$ positive gradient) added to $w$ to push it towards 0  <br>![[Screenshot 2026-08-31 at 9.40.09 AM.png|300]]
	- **Negative gradient**: vice versa <br> ![[Screenshot 2026-08-31 at 9.41.36 AM.png|300]]

#### 3.3. Learning Rate
- Choice of learning rate have a huge impact on the efficiency of implementation of gradient descent

- If $\alpha$ is too small (ie: 0.00001), gradient descent may be very slow <br>![[Screenshot 2026-08-31 at 9.50.50 AM.png|300]]
- If $\alpha$ is too large, may overshoot and fail to converge (never reach minimum) <br>![[Screenshot 2026-08-31 at 9.51.52 AM.png|300]]

- Gradient descent can reach a local minimum with a fixed learning rate: $$w=w-\alpha \frac{d}{dw}J(w)$$
	- Even though $\alpha$ is known as the learning rate, 
	- $\frac{d}{dw}J(w)$ becomes smaller too as it approaches a local minimum 
		- causing the update steps to be smaller even if the learning rate is larger

#### 3.4. Gradient Descent for Linear Regression
$$\begin{align*}
w=w-\alpha\frac{1}{m}\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)} \\
b=b-\alpha\frac{1}{m}\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})\end{align*}$$
- where $f_{w,b}(x^{(i)})$ is the linear regression model: $$f_{w,b}(x^{(i)})=wx^{(i)}+b$$

- **Squared error cost function**: convex function (bowl shape [[#2.3. Visualisation of Cost Function]])
	- Only 1 local minimum, also the global minimum $\to$ always converge to global minimum ($\alpha$ chosen properly) 
