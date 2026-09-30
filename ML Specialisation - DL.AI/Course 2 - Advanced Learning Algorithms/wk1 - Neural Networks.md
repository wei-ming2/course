### 1. Neural Networks
- **Neural network**: algorithm used to make prediction (inference)
	- **also known as**: deep learning algorithm / decision trees

- **NN Origin**: algorithms that mimic how the human brain learns and thinks (to make prediction)

- **Rise of NN**: due more data being available (such as traditional paper receipts being digital) <br> ![[Screenshot 2026-09-17 at 10.05.08 AM.png|500]]
	- **Limitations of Traditional AI techniques (ie: regression)**: performance plateau even with exponentially more data
	- **NN**: as the size of neural network and data becomes larger, the ceiling grows much higher

#### 1.1. Demand Prediction
- **Use case**: predict if a shirt is a top seller so as to better prepare the inventory <br> ![[Screenshot 2026-09-17 at 10.08.16 AM.png|300]]
	- Can be using **logistic regression**: $$a=f(x)=\frac{1}{1+e^{-(wx+b)}}$$
		- $a$: activation
		- **TLDR**: <br> ![[Screenshot 2026-09-17 at 10.13.07 AM.png|300]]

#### 1.2. Building a Neural Network
- **Example**: <br> ![[Screenshot 2026-09-17 at 10.19.16 AM.png|500]]
	- **Layer**: a group of neurons which takes in input as the same/similar features and output a few numbers together
	- **Output layer**: output of the neuron is the prediction
	- **Activations**: neurons sending output values to the neurons downstream from it (affordability, awareness, perceived quality)
	- **TLDR**: 4 numbers (input layer) -> 3 numbers (activation values) -> 1 number (output layer)

- **In practice**: each layer in the middle will have access to every feature from the previous layers; the model itself learns to ignore irrelevant features
	- **Hidden layer**: the interim layers (because in practice, only the input and output is seen)

- **More intuition**: previously for regression, there is (manual) feature engineering (to calculate *area* based on *length* and *width* of land plot)
	- **NN**: learns to engineer its own feature (by itself) to make the problem simpler

- **Architectural of NN**: deciding how many **(1) hidden layers** and **(2) neurons in each layer**

#### 1.3. Face Recognition
- **Facial recognition application**: takes in a picture and identifies the identity of the person in the picture (can be through training a NN)
	- **Input**: 1000 by 1000 pixel image (matrix) of pixel intensity values (rgb)
	- **Example feature layers**: <br> ![[Screenshot 2026-09-22 at 10.04.11 AM.png|500]]
		- **Each layer learns different features, ie**: 
			1. Detects edges / corners
			2. Detects eyes / nose
			3. Detects facial shape

- The process is **strictly progressive** and **hierarchical** (learns small features, before combining them to learn bigger feature). $$\begin{align*}f(\text{pixel})&=\text{edges} \\
  g(\text{edges})&=\text{features} \\
  h(\text{features})&=\text{shapes} \\
  &...\end{align*}$$
	- It is not able to directly predict the face if it does not even understand the nose, or the edges.

#### 1.4. Neural Network Layer
- **Terminology**: 
	- Layer 0: input layer
	- Layer 1: NN (hidden) layer 1
	- Layer 2: NN (hidden) layer 2
	- Layer $n$: output layer

- **Neuron computation in layer 1**: input to activation function to final value $$\begin{bmatrix} 
  \vec{w}_1^{[1]}, b_1^{[1]} \\
  \vec{w}_2^{[1]}, b_2^{[1]} \\
  \vec{w}_3^{[1]}, b_3^{[1]}\end{bmatrix}\to
  \begin{bmatrix} 
  g(\vec{w}_1^{[1]}\cdot\vec{x}+ b_1^{[1]}) \\ 
  g(\vec{w}_2^{[1]}\cdot\vec{x}+ b_2^{[1]}) \\
  g(\vec{w}_3^{[1]}\cdot\vec{x}+ b_3^{[1]})\end{bmatrix}=
  \begin{bmatrix}0.3 \\ 0.7 \\ 0.2\end{bmatrix}$$
	- where $g(z)=\frac{1}{1+e^{-(z)}}$ (or whatever activation function defined, ie: ReLU)
	- $\vec{x}/\vec{a}$ (input): from input data / output vector of previous layer 
	- $\vec{w}$ (weights), $b$ (biases): belongs to the the individual neuron
	- Notation: $\vec{w}_{\text{neuron}}^{[\text{layer}]}$

- **In layer 3**: computes activation using own weights and biases, but activation of previous layer $$a_2^{[3]}=g(\vec{w}_2^{[3]}\cdot \vec{a}^{[2]}+b_2^{[3]})$$

#### 1.5. Making Inference
- **Handwriting digit recognition**: handwriting convert to $8\times8$ grid of rgb values <br> ![[Screenshot 2026-09-23 at 2.34.53 PM.png|300]]
	- Pass through neural network, input to layer 1. Output of layer 1 to layer 2, etc.
	- When it reaches output layer, then the value is compared against threshold for final verdict

- **Forward propagation**: propagating the activations of the neurons (left to right) to get a prediction
- **Back propagation**: used for learning

### 2. Intro to TensorFlow
- **TensorFlow (or PyTorch)**: frameworks for implementing deep learning algorithms
- NN can be applied to different applications

- **Syntax is as follows**: <br> ![[Screenshot 2026-09-24 at 2.29.32 PM.png|300]]
```python
x = np.array([[200.0, 17.0]]) # coffee being roasted at 200 degree for 17 minutes
layer_1 = Dense(units=3, activation='sigmoid')
a1 = layer_1(x) # output of layer 1 with x as input
layer_2 = Dense(units=1, activation='sigmoid')
a2 = layer_2(a1)

if a2 >= 0.5:
	y_hat = 1
else:
	y_hat = 0
```
> **Note**: slight discrepancy in syntax between NumPy and TensorFlow

- **NumPy**: 
```python
x = np.array([[200, 17]]) # [200 17] - 1x2 matrix
x = np.array([[200],  # [200 - 2x1 matrix
			   [17]]) #  17]
x = np.array([200, 17]) # 1D vector
```

- **Tensor to Numpy**:
```python
a1 = layer_1(x)
# tf.Tensor([[0.2, 0.7, 0.3]], shape=(1,3), dtype=float32)
a1.numpy()
# array([[0.2, 0.7, 0.3]], dtype=float32)
```

#### 2.1. Building a Neural Network
- Previously, it relies on making `layer_1`, parsing output (`a1`) to `layer_2` so on so forth. Alternatively:
```python
layer_1 = Dense(units=3, activation='sigmoid', name='layer1')
layer_2 = Dense(units=1, activation='sigmoid')
model = Sequential([layer_1, layer_2])
```

- **Pros**: allow for multiple data points to be trained
```python
x = np.array([[200, 17],
			  [120, 5],
			  [425, 20],
			  [212, 18]])
y = np.array([1,0,0,1])

model.compile(...)
model.fit(x,y)
W1, b1 = model.get_layer("layer1").get_weights() # to get the weights and biases

model.predict(x_new)
```

- **Interpretation of weights (`W1`) and biases (`b1`)**:
```python
W1 = [[-8.94, 0.29, 12.89], 
	  [-0.17, -7.34, 10.79]] 
b1 = [-9.87, -9.28, 1.01]

z_0 = -8.94T -0.17D -9.87 # neuron 0, T: feature 1, D: feature 2
z_1 = 0.29T -7.34D -9.28
z_2 = 12.89T +10.79D +1.01
```
> `z_0` above is normalised, graph below is original unnormalised data
![[Screenshot 2026-09-25 at 1.22.25 PM.png|900]]
- ie: NN layer have 3 neurons, it means there are 3 weights and biases, thus 3 linear lines (2 features means *3 weights for feature 1* and *another 3 for feature 2*)
- sigmoid function: determines what being inside/outside the boundary signify (ie: $z_0 \geq 0.5\to 1$)

#### 2.2. Forward Propagation
```python
def calculate_z(x):
	w1_1 = np.array([1, 2]) # weight
	b1_1 = np.array([-1]) # bias
	z1_1 = np.dot(w1_1, x) + b1_1 # value of linear line
	return z1_1

def neuron(x):
	z1_1 = calculate_z(x)
	a1_1 = sigmoid(z1_1)
	return a1_1
```
- TLDR: steps are first calculate the value using weights and biases, then sigmoid to get overall output of the neuron

- **Full implementation with Python**:
```python
def dense(a_in, W, b):
	units = W.shape[1] # no. of column = no. of neuron
	a_out = np.zeros(units)
	for j in range(units):
		w = W[:, j] # extract the column
		z = np.dot(w, a_in) + b[j]
		a_out[j] = g(z)
	return a_out

def sequential(x):
	a1 = dense(x, W1, b1)
	a2 = dense(a1, W2, b2)
	a3 = dense(a2, W3, b3)
	a4 = dense(a3, W4, b4)
	f_x = a4
	return f_x
```

### 3. Vectorisation
- **Vectorisation**: allows for effective implementation (scaling) of NN (using matrix multiplication)
	- Allowing the CPU/GPU to perform parallel computing

```python
X = np.array([[200, 17]]) # convert to 2d array
W = np.array([[1, -3, 5], # unchanged
			 [-2, 4, -6]])
B = np.array([[-1, 1, 2]]) # 1x3 2d array
```
> 1D array converted to 2D

```python
def dense(A_in, W, B):
	Z = np.matmul(A_in, W) + B
	A_out = g(Z)
	return A_out
```
> Skip the looping code as per [[#2.2. Forward Propagation]]

#### 3.1. Matrix Multiplication
- **Dot product**: array multiplication 1 at a time $$\begin{align*}z&= 
  \begin{bmatrix}1 \\ 2\end{bmatrix}\cdot 
  \begin{bmatrix}3 \\ 4\end{bmatrix} \\
   &=(1\times 3)+(2\times 4) \\
   &=11\end{align*}$$

- **Matrix multiplication**: essentially combines multiple array to a matrix, which are then simultaneously computed $$\begin{align*}z&= 
  \begin{bmatrix}1 & 2\end{bmatrix} 
  \begin{bmatrix}3 & 5 \\ 4 & 6\end{bmatrix} \\
   &=\begin{bmatrix}1(3) +2(4) \\ 1(5)+2(6)\end{bmatrix} \\
   &=\begin{bmatrix}11 & 17\end{bmatrix}\end{align*}$$
	- **Conversion**: $z=\vec{a}\cdot\vec{w}\leftrightarrow\vec{a}^T\vec{w}$

- **Transpose syntax**: 
```python
A = np.array([[1, -1, 0.1],
			  [2, -2, 0.2]])
AT = A.T # transpose
```

- **Matrix multiplication syntax**: 
```python
Z = np.matmul(AT, W)
Z = AT @ W # alternate representation
```
