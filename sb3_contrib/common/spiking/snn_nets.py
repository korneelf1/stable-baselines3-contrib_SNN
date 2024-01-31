import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn as nn
from sb3_contrib.common.spiking.buffers import SNNStates
class SNN_AC(torch.nn.Module):
    def __init__(self, state_shape, action_shape, snn_hidden_size, beta = 0.95, random_beta = False, learn_neuron_params = True):
        super(SNN_AC, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply

        self.snn_hidden_size = snn_hidden_size
        self.state_shape = state_shape
        self.action_shape = action_shape

        beta1 = beta
        beta2 = beta
        beta_crit = beta
        beta_act = beta
        if random_beta:
            beta1 = torch.rand(snn_hidden_size)
            beta2 = torch.rand(snn_hidden_size)
            beta_crit = torch.rand(1)
            beta_act = torch.rand(action_shape)
        
        self.lin1 = nn.Linear(state_shape, snn_hidden_size)
        self.lif1 = snn.Leaky(beta = beta1, spike_grad=self.spike_grad, learn_beta=learn_neuron_params)

        self.lin2 = nn.Linear(snn_hidden_size, snn_hidden_size)
        self.lif2 = snn.Leaky(beta = beta2, spike_grad=self.spike_grad, learn_beta=learn_neuron_params)

        self.critic_linear = nn.Linear(snn_hidden_size, 1)
        self.lif_crit = snn.Leaky(beta = beta_crit, spike_grad=self.spike_grad, learn_beta=learn_neuron_params)
        self.actor_linear = nn.Linear(snn_hidden_size, action_shape)
        self.lif_act = snn.Leaky(beta = beta_act, spike_grad=self.spike_grad, learn_beta=learn_neuron_params)

        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif2.init_leaky()
        self.mem_crit = self.lif_crit.init_leaky()
        self.mem_act  = self.lif_act.init_leaky()


        self.train()

    def init_mem(self):
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif2.init_leaky()
        self.mem_crit = self.lif_crit.init_leaky()
        self.mem_act  = self.lif_act.init_leaky()

    def forward(self, inputs, states):
        self.mem1 = states.pi_potentials[0]
        self.mem2 = states.pi_potentials[1]
        self.mem_act = states.pi_potentials[2]
        self.mem_crit = states.vi_potentials[0]
        for i in range(len(inputs)):
            x = inputs.float()
            x = self.lin1(x)
            x, self.mem1 = self.lif1(x, self.mem1)
            x = self.lin2(x)
            x, self.mem2 = self.lif2(x, self.mem2)

        value = self.critic_linear(x)
        value, self.mem_crit = self.lif_crit(value, self.mem_crit)
        action = self.actor_linear(x)
        action, self.mem_act = self.lif_act(action, self.mem_act)
        return [value, action], SNNStates(pi_potentials=(self.mem1,self.mem2,self.mem_act), vf_potentials=(self.mem_crit))
    
class SNN(nn.Module):
    "creates SNN with n hidden layer and output layer"
    def __init__(self, input_size, output_dim, snn_hidden_size, n_hidden, beta = 0.95, random_beta = True, learn_neuron_params = True):
        super(SNN, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply

        self.hidden_size = snn_hidden_size
        self.input_size = input_size
        self.output_dim = output_dim
        self.num_layers = n_hidden + 1
        betas_hidden = []
        betas_out = 0.9
        if random_beta:
            for i in range(n_hidden):
                betas_hidden.append(torch.rand(self.hidden_size))
            betas_out = torch.rand(output_dim)
        else:
            betas_hidden = [beta]*n_hidden
            betas_out = beta

        # create hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_neurons = nn.ModuleList()

        self.hidden_mems = []
        if n_hidden > 0:
            self.hidden_layers.append(nn.Linear(input_size, self.hidden_size))
      
            for i in range(n_hidden):
                self.hidden_neurons.append(snn.Leaky(beta = betas_hidden[i], spike_grad=self.spike_grad, learn_beta=learn_neuron_params))
            for i in range(n_hidden-1):
                self.hidden_layers.append(nn.Linear(self.hidden_size, snn_hidden_size))

            self.output_layer = nn.Linear(self.hidden_size, output_dim)
        else:
            self.output_layer = nn.Linear(input_size, output_dim)
        self.output_neuron = snn.Leaky(beta = betas_out, spike_grad=self.spike_grad, learn_beta=learn_neuron_params)
        
        for neurons in self.hidden_neurons:
            self.hidden_mems.append(neurons.init_leaky())
        self.output_mem = self.output_neuron.init_leaky()

        self.train()

    def init_mem(self):
        for neurons in self.hidden_neurons:
            self.hidden_mems.append(neurons.init_leaky())
        self.output_mem = self.output_neuron.init_leaky()

    def forward(self, inputs, states, return_all = True):
        '''
        inputs: tensor of shape (n_steps,n_parallels, input_size)
        states: list of tensors of shape (nr_layers, n_steps, hidden_size) hmm should it not be (n_parallels, nr_layers, n_steps, hidden_size)?
        (n_layers, n_seq, dim)
        states in is for the correct input at start of sequence
        
        output: tensor of shape (n_steps, n_parallels, output_dim)
        output states is the tensor containing final hidden states for each layer (nr_layers, n_parallels, hidden_size)
        '''
        # assert inputs.shape[0] == states.shape[0]
        assert inputs.shape[1] == states.shape[1]
        self.hidden_mems = [len(self.hidden_layers)*[]]


        outputs = []
        output_mems = []
        for j in range(len(self.hidden_layers)):
            output_mems.append(states[j,:,:])
        output_mems.append(states[-1,:,:])

        for i in range(inputs.shape[0]):
            x = inputs.float()[i,:,:]
            
            for j in range(len(self.hidden_layers)):
                x = self.hidden_layers[j](x)
                x, output_mems[j] = self.hidden_neurons[j](x, output_mems[j])

            x = self.output_layer(x)
            x, output_mems[-1] = self.output_neuron(x, output_mems[-1])
            outputs.append(x) # gotta figure out how to pass the whole sequence to the mlp extractor if necessary

        output_mems = torch.stack(output_mems, dim=0)
        outputs = torch.stack(outputs, dim=0)

        # ISSUE: only processed output is returned, we need intermediate steps for the snn mlp extractor...
        if return_all:
            return outputs, output_mems
        return outputs, output_mems

if __name__ == '__main__':
    import torch.optim as optim
    import torch.nn.functional as F
    # Define your loss function and optimizer


    # Set random seed for reproducibility
    torch.manual_seed(42)

    true_targets = torch.ones(5, 1, 10)  # All zeros for debugging
    # test
    model = SNN(4, 10, 10, 2)

    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)


    inputs = torch.rand(5, 1, 4)
    print('Sequential:')
    outputs1, states1 = model(inputs, torch.zeros(3, 1, 10))
    loss = criterion(outputs1, true_targets)
    # Backward pass
    optimizer.zero_grad()
    loss.backward() # for some reason last layers not backpropping
    
    # Print gradie  nts for debugging
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'Gradient - {name}: {param.grad.mean()}')
    


    print('-------------------')
    model.init_mem()

    print('Sequential with states:')
    states2 = torch.zeros(3, 1, 10)
    states_lst = [states2]
    loss = 0
    for i in range(5):
        outputs2, states2 = model(inputs[i,:,:].unsqueeze(0), states2)
        states_lst.append(states2)
        loss += criterion(outputs1, true_targets)

        # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Print gradients for debugging
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'Gradient - {name}: {param.grad.mean()}')


    print('-------------------')
    model.init_mem()
    states3 = torch.stack(states_lst, dim=0)
    print('Shuffled but with states:')
    # pass shuffled but with states
    indices = torch.randperm(5)
    print(states1-states2)
    print(indices)
    for i in indices:
        outputs3, states3 = model(inputs[i,:,:].unsqueeze(0), states_lst[i])
        loss += criterion(outputs1, true_targets)

        # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Print gradients for debugging
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'Gradient - {name}: {param.grad.mean()}')
    


    print('-------------------')
    print()