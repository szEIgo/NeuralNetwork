# Working with tensorflow
## Dynamical Graphs, Deep learning & Neural Networks (by Google)
pip3 install tensorflow

## Dynamical graph representation 
Dynamical graph representation of tensorflow learning to identify numbers from images


## Run our code

python3 tutorial.py

## Softmax And Cross-entropy

### Softmax
 
A normalized exponential function -  that is, a probability distribution over K different possible outcomes
![alt tag](https://github.com/szEIgo/NeuralNetwork/blob/master/math1.png)

### Cross-Entropy

It changes the bias and the weights, and then compare the output value with the correct value, to minimize the differences.

![alt tag](https://github.com/szEIgo/NeuralNetwork/blob/master/img2.png)

### Needed variables:
- input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
 - X = tf.placeholder(tf.float32, [None, 28, 28, 1])
 - correct answers will go here
 - Y_ = tf.placeholder(tf.float32, [None, 10])
 - weights W[784, 10]   784=28*28
 - W = tf.Variable(tf.zeros([784, 10]))
 - biases b[10]
 - b = tf.Variable(tf.zeros([10]))


- Iterations - GDO 	  - Precision
- 2000       - 0.003 	- 0.9221
- 5000       -  0.003	- 0.9242
- 2000 	     - 0.004	 - 0.9244
- 2000	      - 0.004	 - 0.9251
- 2000 	     - 0.005	 - 0.9218
- 5000 	     - 0.005	 - 0.9257
- 2000 	     - 0.006	 - 0.9213
- 5000       - 0.006	 - 0.9254

![alt tag](https://github.com/szEIgo/NeuralNetwork/blob/master/math2.png)

That didn't change much.
my CPU could not handle visualization fast enough to test the precisions up against each other.
![alt tag](https://github.com/szEIgo/NeuralNetwork/blob/master/img1.png)

## Five layers sigmoid

4 layers of sigmoid neurons(200, 100, 60, 30)
1 layer of Softmax neurons(10)

Last layer uses softmax aswell.

 - Iterations  - AO     - Precision
 - 10000       - 0.003 	- 0.9809
 - 20000       - 0.003  - 0.9873
 - 10000       - 0.005 	- 0.9795

 Neurons changed to see output

 - Neurons             - AO    - Precision - iterations
 - (100, 50, 20, 10)   - 0.005 - 0.9724    - 10000
 - (1000, 500, 200, 10)- 0.005 - 0.9678    - 1000



![alt tag](https://github.com/szEIgo/NeuralNetwork/blob/master/img3.png)





