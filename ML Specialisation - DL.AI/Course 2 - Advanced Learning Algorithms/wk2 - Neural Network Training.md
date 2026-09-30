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



