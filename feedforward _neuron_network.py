import numpy as np
class NeuronNetwork():
    def __init__(self):
        self.w1 =np.random.normal()
        self.w2 =np.random.normal()
        self.w3 =np.random.normal()
        self.w4 =np.random.normal()
        self.w5 =np.random.normal()
        self.w6 =np.random.normal()
        
        self.b1 =np.random.normal()
        self.b2 =np.random.normal()
        self.b3 =np.random.normal()

        self.h1 = AI_Neuron([self.w1, self.w2], self.b1)
        self.h2 = AI_Neuron([self.w3, self.w4], self.b2)
        self.o1 = AI_Neuron([self.w5, self.w6], self.b3)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 =self.o1.feedforward(np.array([out_h1,out_h2]))
        return out_o1
    
    def deriv_sigmoid(self, out_neuron):
        return out_neuron * (1-out_neuron)
    
    def mse_loss(self,y_true,y_pred):
        return ((y_true - y_pred) ** 2).mean()
    
    def train(self,data,all_y_trues):
        learn_rate = 0.1
        epochs = 1000
        for epoch in range(epochs):
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward,1,data)
                loss= self.mse_loss(all_y_trues,y_preds)
                print('Epoch %d loss: %.3f' % (epoch,loss))
            for x,y_true in zip(data, all_y_trues):
                y_pred= self.feedforward(x)
               
                d_L_d_ypred = -2 * (y_true - y_pred)

                d_ypred_d_w5 = self.h1.feedforward(x) * self.deriv_sigmoid(y_pred)
                d_ypred_d_w6 = self.h2.feedforward(x) * self.deriv_sigmoid(y_pred)
                d_ypred_d_b3 = self.deriv_sigmoid(y_pred)

                d_ypred_d_h1 = self.w5 * self.deriv_sigmoid(y_pred)
                d_ypred_d_h2 = self.w6 * self.deriv_sigmoid(y_pred)

                
                d_h1_d_w1 = x[0] * self.deriv_sigmoid(self.h1.feedforward(x))
                d_h1_d_w2 = x[1] * self.deriv_sigmoid(self.h1.feedforward(x))
                d_h1_d_b1 = self.deriv_sigmoid(self.h1.feedforward(x))

          
                d_h2_d_w3 = x[0] * self.deriv_sigmoid(self.h2.feedforward(x))
                d_h2_d_w4 = x[1] * self.deriv_sigmoid(self.h2.feedforward(x))
                d_h2_d_b2 = self.deriv_sigmoid(self.h2.feedforward(x))
                
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                self.h1.weights = np.array([self.w1, self.w2])
                self.h1.bias = self.b1
                
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                self.h2.weights = np.array([self.w3, self.w4])
                self.h2.bias = self.b2

                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
                self.o1.weights = np.array([self.w5, self.w6])
                self.o1.bias = self.b3


    

class AI_Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def feedforward(self, x):
        total = np.dot(self.weights, x) + self.bias
        return self.sigmoid(total)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self,x):
        total= np.dot(self.weights, x) + self.bias
        return self.sigmoid(total)

if __name__ == '__main__':
    data = np.array([
        [-2,-1],
        [24,13],
        [11,8],
        [-12,-10],
    ])

    all_y_trues= np.array([
        1,
        0,
        0,
        1,
    ])

    network= NeuronNetwork()
    network.train(data,all_y_trues)

    karina = np.array([-7,-3])
    mikola= np.array([20,2])
    person = np.array([2,-2])
    person_me = np.array([6,7])

    print('Karina: %.3f' % network.feedforward(karina))
    print('Mikola: %.3f' % network.feedforward(mikola))
    print('Person: %.3f' % network.feedforward(person))
    print('Me: %.3f' % network.feedforward(person_me))