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
- **Handwriting digit recognition**: 

