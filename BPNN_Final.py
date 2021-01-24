# This program works with 2 input, 2 hidden and 1 output layer.
# Author: Tanveer Ahmed Khan
# BSCIS(18-22)- 5th Semester PIEAS
# AI Assignement - BPNN


from numpy import *
import numpy as np
import random



#   Required variables and constants

e =2.71828
Target_output=0.0
d=np.array([0.0])
Y=np.zeros((2,1))
#   Assume
alpha=0.2
n=0.6
#   Hidden layer weights
delta_W=np.zeros((2,1))
weights_e=np.zeros((2,1))

# Input layer weights
delta_v=np.zeros((2,2))

d_star=np.zeros((2,1))
x=np.zeros((2,2))
output_of_input_layer=np.zeros((2,1))
weights_of_input_layer=np.zeros((2,2))
weights_of_hidden_layer=np.zeros((2,1))
input_for_hidden_layer=np.zeros((2,1))
output_of_hidden_layer=np.zeros((2,1))
i=0


#   Generating weights for input layer
for i in range(0,2):
    for j in range(0,2):
        num=random.uniform(-1.0,1.0)
        num=round(num,1)
        weights_of_input_layer[i][j]=num

#   Generating weights for hidden layer

for i in range(0,2):
    for j in range(0,1):
        num=random.uniform(-1.0,1.0)
        num=round(num,1)
        weights_of_hidden_layer[i][j]=num



#   input input layer values

for i in range(0,2):
    for j in range(0,1):
        output_of_input_layer[i][j]=float(input(f"Enter Input for {i+1} Neuron: "))
    # output_of_I[1][0]=float(input("Enter Input for 2nd Neuron: "))
Target_output=float(input("Enter Target Value: "))
itr=int(input("Enter number of iterations: "))


#   Printing Initial Weights
print("\n*****Initial Weights....*****")
print("Input Layer Weights\n",weights_of_input_layer)
print("Hidden layer Weights\n",weights_of_hidden_layer)




#   ********************While Start**********************


while i<itr:
    #   Transpose of weights of input
    weights_of_input_layer_transpose=transpose(weights_of_input_layer)

    #   Working at Hidden Layers

   
    input_for_hidden_layer=np.dot(weights_of_input_layer_transpose,output_of_input_layer)
    h1=input_for_hidden_layer[0][0]
    h2=input_for_hidden_layer[1][0]
    h1_out=(1/(1+(e**(-h1))))
    h2_out=(1/(1+(e**(-h2))))
    



    #   Working at Output Layer

    output_of_hidden_layer[0][0]=h1_out
    output_of_hidden_layer[1][0]=h2_out
    weights_of_hidden_layer_transpose=transpose(weights_of_hidden_layer)
    input_of_output_layer=np.dot(weights_of_hidden_layer_transpose,output_of_hidden_layer)
  

    output_of_output_layer=1/(1+(e**(-(input_of_output_layer))))


    #   finding error


    error=(Target_output-output_of_output_layer)**2


    #   finding d

    d=(Target_output-output_of_output_layer)*(output_of_output_layer)*(1-output_of_output_layer)
    dt=transpose(d)



    #   Calculating Y

    Y=np.dot(output_of_hidden_layer,dt)
    Y=Y.reshape(2,1)



    #   change in weights of hidden layer

    delta_W=((alpha*(delta_W))+(n*(Y)))



    #   Calculating Hidden layer weights e
    weights_e=weights_of_hidden_layer*d
    weights_e=weights_e.reshape(2,1)


    #   Calculating d star

    d_star=[
        [(weights_e[0][0])*(output_of_hidden_layer[0][0])*(1-output_of_hidden_layer[0][0])],
        [(weights_e[1][0])*(output_of_hidden_layer[1][0])*(1-output_of_hidden_layer[1][0])]
    ]
    d_star_transpose=transpose(d_star)


    #   Calculating x

    x=output_of_input_layer*d_star_transpose


    #   Calculating change in Input Layer Weights

    delta_v=(alpha*(delta_v)+n*(x))

    #   Updating Weights

    weights_of_input_layer=weights_of_input_layer+delta_v
    weights_of_hidden_layer=weights_of_hidden_layer+delta_W

    i=i+1
#   ********************While end**********************


#   Printing Results

print("\n\n*********RESULTS*********\n")
print(f'After {i} iterations.')
print('Target Output is: ',Target_output)
print('Output is: ',output_of_output_layer)
print('Error: ',error)

#   Printing Final Weights
print("\n*****Final Weights....*****\n")
print("Input Layer Weights\n",weights_of_input_layer)
print("Hidden layer Weights\n",weights_of_hidden_layer)