### 1. Advice for Applying Machine Learning
#### 1.1. Debugging a Learning Algorithm
- Eg: implemented a regularised linear regression on housing price [[wk3 - Classification#3.2. Cost Function with Regularisation]]
	- But it still makes unacceptably large errors in prediction. You could try:
		- Getting more training examples
		- Additional / smaller set of features
		- Increasing / decreasing $\lambda$

- **Machine learning diagnostic**: test to gain insight into what is/isn't working

#### 1.2. Evaluating a Model
- Eg: [[wk3 - Classification#3. The Problem of Overfitting]]
	- Model with 5 data point but 4-order polynomial
	- Fits the data extremely well, but it fails to generalise to new examples.

##### 1.2.1. Train-Test Split
- Rather than training on the full data, train only on 70% of the original dataset, and use the remainder 30% to evaluate

- **Minimising cost function $J(\vec{w},b)$**: using train (average of cost and regularisation) $$J(\vec{w},b)=\left[\frac{1}{2m_\text{train}}\sum_{i=1}^{m_{\text{train}}}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2+\frac{\lambda}{2m_\text{train}}\sum_{j=1}^nw_j^2\right]$$
- **Compute test error $J_\text{test}(\vec{w},b)$**: using test $$J_\text{test}(\vec{w},b)=\frac{1}{2m_\text{test}}\left[\sum_{i=1}^{m_{\text{test}}}\left(f_{\vec{w},b}(\vec{x}^{(i)}_\text{test})-y^{(i)}_\text{test}\right)^2\right]$$

- Linking back to [[#1.2. Evaluating a Model]], the $J_\text{train}$ will be low, but $J_\text{test}$ will be high
	- Similarly applies to logistic regression (with different cost and error function)
		- or count the fraction of train and test set that was misclassified (another way to evaluate the model)

#### 1.3. Model Selection
1. **Bad practice**: brute force
2. **Better**: cross validation

##### 1.3.1. Brute Force

| Degree | Model                                         | Evaluate                         |
| ------ | --------------------------------------------- | -------------------------------- |
| 1      | $f_{\vec{w},b}(\vec{x})=w_1x+b$               | $J_\text{test}(w^{(1)},b^{(1)})$ |
| 2      | $f_{\vec{w},b}(\vec{x})=w_1x+w_2x^2+b$        | $J_\text{test}(w^{(2)},b^{(2)})$ |
| 3      | $f_{\vec{w},b}(\vec{x})=w_1x+w_2x^2+w_3x^3+b$ | $J_\text{test}(w^{(3)},b^{(3)})$ |
| ...    | ...                                           | ...                              |
- Look through lots of model with different degree and select model with lowest cost

##### 1.3.2. Cross Validation
- Split data into:
	- Training set (60%)
	- Cross validation set (20%)
	- Test set (20%)

- **Cross validation error**: $$J_\text{cv}(\vec{w},b)=\frac{1}{2m_\text{cv}}\left[\sum_{i=1}^{m_{\text{cv}}}\left(f_{\vec{w},b}(\vec{x}^{(i)}_\text{cv})-y^{(i)}_\text{cv}\right)^2\right]$$
	- Essentially the same as training / test except calculated using cv set thus $J_\text{cv}$

| Degree | Model                           | Evaluate                       |
| ------ | ------------------------------- | ------------------------------ |
| 1      | $f_{\vec{w},b}(\vec{x})=w_1x+b$ | $J_\text{cv}(w^{(1)},b^{(1)})$ |
| ...    | ...                             | ...                            |
- Similar to *brute force*, except $J_\text{test}\to J_\text{cv}$
	- Training with **training set**
	- Choose the model (and tune) base on **cross validation set**
	- Estimate generalisation error using **test set**

##### 1.3.3. Feature Scaling
- Perform feature scaling to help model converge faster (especially when input features have a wide range of values - ie: 2 to 12 million) 

- **$z$-normalisation**: each value is the number of SD away from mean $$z=\frac{x-\mu}{\sigma}$$

### 2. Bias and Variance
- **Recap**: [[wk3 - Classification#3. The Problem of Overfitting]]

| Bias / Variance  | Type                         | $J_\text{train}$ | $J_\text{cv}$ |
| ---------------- | ---------------------------- | ---------------- | ------------- |
| High bias        | underfit<br>(low level poly) | high             | high          |
| **"Just right"** | **generalise**<br>(middle)   | **low**          | **low**       |
| High variance    | overfit<br>(high level poly) | low              | high          |
- High bias and high variance: 
	- $J_\text{train}$ will be high
	- $J_\text{cv} >>J_\text{train}$

#### 2.1. Regularisation
- **Recap**: ![[wk3 - Classification#3.2. Cost Function with Regularisation]]

- **Choosing the regularisation parameter $\lambda$**: 
	1. Try $\lambda=0$
	2. Find the values of $\vec{w},b$ that minimise the cost function $J$: $\underset{\vec{w},b}{\min}J(\vec{w},b)$ 
	3. Get the optimal parameter of the model from (2): $w^{<1>},b^{<1>}$ 
	4. Get the cross validation error of the model: $J_\text{cv}(w^{<1>},b^{<1>})$
		- Try $\lambda=0.01$ (repeat step 1 to 4)
		- Try $\lambda =0.02$
		- ...
		- Plot $J_\text{cv}(\vec{w},b)$ and $J_\text{train}(\vec{w},b)$ against $\lambda$: <br>![[Screenshot 2026-10-27 at 9.24.27 AM.png|500]]
	5. After selecting the optimal $\lambda$ (lowest $J_\text{cv}$), find the $J_\text{test}(w^{<\text{best}>},b^{<\text{best}>})$

#### 2.2. Establishing a Baseline Level of Performance
- **Example**: speech recognition model
	- Training error $J_\text{train}:10.8\%$
	- Cross validation error $J_\text{cv}:14.8\%$
	- **Human level performance**: $10.6\%$
		- Due to background noise, poor speaker quality, etc
		- If humans, the trainer only gets 10.6%, unlikely that model is any better

- To judge if a training error is high, to see if *training error* is much higher than *human level of performance*
	- In the example, the model performs extremely near to *human level performance*, but further to *cross validation* (thus it is more of a variance/overfit than a bias/underfit problem)

- Alternatively: performance can be measured in comparison to
	- Competing algorithms performance
	- Guess based on prior experience

#### 2.3. Learning Curve
- **Intuition**: $J_\text{cv}$ decreases, but $J_\text{train}$ increases with more training data <br>![[Screenshot 2026-10-29 at 9.34.12 AM.png|300]]
	- $J_\text{train}$ increases because it gets harder to fit everything perfectly as the number of data point increases <br>![[Screenshot 2026-10-27 at 11.55.53 AM.png|300]]

- **High bias learning curve**: underfit (opposite of typical understanding) <br>![[Screenshot 2026-10-29 at 9.34.12 AM.png|300]]
	- As training size grows, there is not that much to change, thus error flattens out after a while <br>![[Screenshot 2026-10-28 at 6.22.23 PM.png|100]]
	- $J_\text{cv}$ comes down but also flattens (because there is only so much that can be captured)

- **High variance learning curve**: overfit <br>![[Screenshot 2026-10-29 at 9.37.32 AM.png|300]]
	- More training data is likely to help reduce error <br>![[Screenshot 2026-10-29 at 9.38.27 AM.png|100]]

#### 2.4. Deciding what to try next
- **Continuation of**: ![[#1.1. Debugging a Learning Algorithm]]
	- The 6 items essentially tries to fix either a *high variance* or a *high bias* problem

| Method                                           | Addresses     |
| ------------------------------------------------ | ------------- |
| More training example                            | High variance |
| Smaller set of feature<br>(remove $x^3,x^4,...$) | High variance |
| Additional features                              | High variance |
| Adding polynomial                                | High bias     |
| Increasing $\lambda$                             | High variance |
| Decreasing $\lambda$                             | High bias     |