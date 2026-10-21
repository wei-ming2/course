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

- **Step**: similarly to brute force, except the degree with the 