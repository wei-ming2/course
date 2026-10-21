### 1. Neural Network Training
#### 1.1. TensorFlow Implementation
- **Defining the NN**:
```python
model = Sequential([
	Dense(units=25, activation='sigmoid'),
	...
])
```

- **Defining the loss and compiling**: 
```python
from tensorflow.keras.losses import BinaryCrossentropy

model.compile(loss=BinaryCrossentropy())
model.fit(X, Y, epochs=100)
```

#### 1.2. Training Details
- **Logistic regression training steps**:
	1. Specify how to compute output given input $x$ and parameters $w,b$
	2. Specify loss and cost (average loss)
	3. Train on data to minimise $J(\vec{w},b)$ (using gradient descent)

- **NN training steps**: 
	1. Define the NN architectural
	2. Compile the model with a loss function
	3. Fit to try to minimise the cost function of the parameter

- **Binary cross entropy loss function**: essentially the same as ![[wk3 - Classification#2.2. Simplified Loss Function]]
	- Parse the loss function to TF and it knows to minimise the cost function

- **Step 3**: minimise the cost function by computing derivative for gradient descent using 'back-propagation' (`model.fit(...)`)

### 2. Activation Functions
- **Activation function**: function that is applied to the inputs to produce the final output of a neuron

- **Previously what was learnt / used**: building a NN by taking logistic regression and stringing the logistic regression units together
- **Using other activation functions**: allow for more flexibility, much more powerful
	- **Example**: *demand prediction* given *price*, *shipping cost*, *marketing*, and *material*

| Feature               |              | Awareness      |                    |
| --------------------- | :----------- | :------------- | :----------------- |
| Binary Classification | 0            |                | 1                  |
| In Reality            | No awareness | Some awareness | Complete awareness |

- **Rectified Linear Unit (ReLU)**: takes on 0 or any non-negative number <br> ![[Screenshot 2026-09-29 at 10.16.39 AM.png|300]]
	- Ie: *awareness* should be a non-negative number

- **Examples of activation functions**: <br>![[Screenshot 2026-09-29 at 10.22.30 AM.png]]

#### 2.1. Choosing Activation Function
- Choice of activation function for **hidden layer**, **output layer**

| Activation Function | Use Case (Output layer)              | Example               |
| ------------------- | ------------------------------------ | --------------------- |
| Sigmoid             | Binary classification: output 0 or 1 | Predict pass or fail  |
| Linear activation   | Regression: can be any value         | Predict stock price   |
| ReLU                | Regression: only non-negative value  | Predict housing price |

- **Hidden layer**: ReLU (most common)
	- **Pros**: 
		- Simpler to compute $\text{max}(0,z)$
		- Gradient descent is faster (flat at negative vs flat at both extreme)

- **Rationale for activation function**: allows for modelling of more complex relationship
	- **Example**: if all activation function is linear ($g(z)=z$), it essentially is no different than a linear regression function <br>![[Screenshot 2026-09-29 at 10.52.40 AM.png|300]]
		- **Neuron 1**: $a^{[1]}=z=w_1^{[1]}x+b_1^{[1]}$
		- **Neuron 2**: only a single $x$-term in the whole equation $$\begin{align*}a^{[2]}
		  &=w_1^{[2]}a^{[1]}+b_1^{[2]} \\
			  &=w_1^{[2]}(w_1^{[1]}x+b_1^{[1]})+b_1^{[2]} \\
			  &=wx+b\end{align*}$$

### 3. Multiclass Classification
- Refers to classification problems with more than just 2 (0 or 1) possible output values
	- Ie: handwriting classification (numbers 0 to 9), identifying type of defect, etc <br>![[Screenshot 2026-09-29 at 11.26.27 AM.png|500]]
		- Rather than separating data into 2 categories with a single decision boundary, multiclass assigns each item to one of multiple possible classes

#### 3.1. Softmax 
- **Softmax**: binary classification algorithm, to the multiclass classification context  (generalisation of logistic regression)
	- Logistic regression: if $P(y=1|\vec{x})=0.71$, $P(y=0|\vec{x})=1-0.71=0.29$

- Softmax regression: 4 possible output <br>![[Screenshot 2026-09-29 at 1.24.08 PM.png|500]]
	- $a_1$: algorithm's estimate of the chance of $y$ being equal to 1, given input feature $x$ $$a_1=\frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}+e^{z_4}}$$
	- $a_2$: estimate of the chance of $y=2$, etc 
		- predicting the chances of $Y$ being $1,2,3,4$ given a set of input features $X$

- **Cost function**:  $$loss(a_1,...,a_N,y)=\begin{cases}
  -\log a_1 \text{ if }y=1 \\
  -\log a_2 \text{ if }y=2 \\
  \vdots \\
  -\log a_N \text{ if }y=N \\
  \end{cases}$$
	- Essentially, if $y=2$, $a_2$ should be as close to $1$, and $a_1,a_3,...$ being small (else cost is huge) <br>![[Screenshot 2026-09-29 at 2.08.34 PM.png|300]]

#### 3.2. Neural Network with Softmax Output
- **Previously, 1 or 0**: <br>![[Screenshot 2026-09-29 at 2.12.28 PM.png|300]]

- **Updated**: <br>![[Screenshot 2026-09-29 at 2.13.39 PM.png|300]]
```python
model = Sequential([
	Dense(units=25, activation='relu'),
	Dense(units=15, activation='relu'),
	Dense(units=10, activation='softmax')
])

model.compile(loss=SpareCategoricalCrossentropy())
```

- **Note**: numerical roundoff error (due to computer's storage of digits)
```python
x1 = 2 / 1000
print(f"{x1:.18f}") # 0.00200...

x2 = 1 + (1/1000) - (1- 1/1000) # mathematically equivilant
print(f"{x2:.18f}") # 0.00199...
```

- **To Fix (Idea)**: parses the full working to the loss function rather than the intermediary value $$-\log a_{10}\to -\log \frac{e^{z_1}}{e^{z_1}+...+e^{z_{10}}}$$
```python
Dense(units=10, activation='linear') # model: activation no longer softmax

model.compile(loss=SpareCategoricalCrossEntropy(from_logits=True)) # loss: from_logits

logits = model(X) # predict: no longer probabilities a1 to a10, instead z1 to z10
f_x = tf.nn.softmax(logits)
```

- **Multi-label Classification**: for each set of $x$ input, there can be a set of $y$ outputs (or labels)

#### 3.3. Code Example
```python
model = Sequential(
	[
		Dense(2, activation='relu', name='L1'),
		Dense(4, activation='linear', name='L2')
	]
)
```
- Simple 2-layer network for multi-class classification: <br>![[Screenshot 2026-10-12 at 10.47.11 AM.png|300]]

- **How the model works**: <br>![[Screenshot 2026-10-12 at 10.52.30 AM.png|500]]

| Neuron 0 | Neuron 1 | Classification |
| -------- | -------- | -------------- |
| Inactive | Inactive | c0             |
| Active   | Inactive | c2             |
| Inactive | Active   | c1             |
| Active   | Active   | c3             |

### 4. Advanced NN Concepts
#### 4.1. Advanced Optimisation
- **Intuition**: *gradient descent* tend to slowly converge to the minimum in the same direction for a bit
	- How can we optimise the algorithm for greater training speed?

- **Adaptive Moment Estimation (Adam)**: automatically adjusts the learning rate ($\alpha$)<br>![[Screenshot 2026-10-12 at 2.23.20 PM.png|300]]![[Screenshot 2026-10-12 at 2.26.39 PM.png|300]]
	- If $w_j$ (or $b$) keeps moving in same direction: increase $\alpha_j$
	- If $w_j$ (or $b$) keeps oscillating: reduce $\alpha_j$

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
```

#### 4.2. Additional Layer Types
- **Convolutional layer**: each neuron only looks at some parts of the previous layer's output (Dense layer looks at every previous neuron) <br>![[Screenshot 2026-10-12 at 2.33.05 PM.png|200]]
	- **Benefits**: 
		1. Faster computation
		2. Need less training data / less prone to overfitting

### 5. Back Propagation
#### 5.1. Derivative
- Eg: cost function $J(w)=w^2$
	- Derivative is: $2w$ (meaning $w^2$ is changing at a rate of $2w$)

- `sympy` package:
```python
J, w = sympy.symbols("J,w")
J = w**3
dJ_dw = sympy.diff(J, w) # 3w^2
dJ_dw.subs([(w, 2)]) # substitute w as 2 = 12
```

#### 5.2. Computation Graph
- For a neuron, there is an input ($\vec{x}$) and an output ($a$)
	- If $a=wx+b$, where $w=2, b=8, x=-2, y=2$: computation graph essentially breaks down a series of working (to get to $J$) into individual step <br>![[Screenshot 2026-10-19 at 9.00.27 AM.png|700]]
		- Forward propagation: left to right computation
		- Back propagation: right to left

##### 5.2.1. Back Propagation
- Let's say that: $d$ increases by $\varepsilon$, $J$ increases by $2\varepsilon$ $$\begin{align*}J_{\text{new}}&=\frac{1}{2}(2+\varepsilon)^2 \\
  &=\frac{1}{2}(4+4\varepsilon+\varepsilon^2) \\
  \Delta J&=2\varepsilon+\frac{1}{2}\varepsilon^2 \\
  &\approx 2\varepsilon \hspace{0.2cm}(\text{if }\varepsilon\text{ is small})\end{align*}$$

- Connecting back to [[#5.2. Computation Graph]]: $$\begin{align*}\frac{\partial J}{\partial a}&=\frac{\partial J}{\partial d}\frac{\partial d}{\partial a} \\
  &=d\cdot 1 \\
  &=2\end{align*}$$
	- Partial derivative of $J=\frac{1}{2}d^2$: $$\frac{\partial J}{\partial d}=d$$
	- Partial derivative of $d=a-y$: $$\frac{\partial d}{\partial a}=1$$

- Similarly (using chain rule for the previous partial derivatives): $$\begin{align*}\frac{\partial J}{\partial b}&=\frac{\partial J}{\partial a}\frac{\partial a}{\partial b} \\
  &=2\cdot 1 \\
  &= 2\end{align*}$$
- and:  $$\begin{align*}\frac{\partial J}{\partial c}&=\frac{\partial J}{\partial a}\frac{\partial a}{\partial c} \\
  &= 2\cdot 1 \\
  &= 2\end{align*}$$
- finally: $$\begin{align*}\frac{\partial J}{\partial w}&=\frac{\partial J}{\partial c}\frac{\partial c}{\partial w} \\
  &= 2\cdot x \\
  &= -4\end{align*}$$
	- **Back propagation**: calculates the gradients of the loss $J$ wrt intermediate variables $(d,a,b,c)$ and model parameters by applying the chain rule backwards through the computation graph.
		- The resulting gradients can then by used by an optimiser to update weights and biases to reduce the loss. (Back propagation: gradient -> optimiser: reduce overall loss)

- Example: to verify, when $w\uparrow0.001$ $$\begin{align*}J&=\frac{1}{2}((wx+b)-y)^2 \\
  &=\frac{1}{2}((2.001\times-2+8)-2)^2 \\
  &= 1.996002\approx 1.996\end{align*}$$
	- $J\downarrow 0.004$, decrease is approximately -4 of $w$ (which is the gradient and the relationship)
