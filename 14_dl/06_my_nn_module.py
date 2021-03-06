import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, N_I, N_H, N_O):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(N_I, N_H)
        self.linear2 = torch.nn.Linear(N_H, N_O)

    def forward(self, x):
        h = self.linear1(x)
        h_r = h.clamp(min=0)
        y_p = self.linear2(h_r)
        return y_p

EPOCHS = 300
M = 64
N_I = 1000
N_H = 100
N_O = 10
LEARNING_RATE = 1.0e-04

# create random input and output data
x = torch.randn(M, N_I)
y = torch.randn(M, N_O)

# define model
model = TwoLayerNet(N_I, N_H, N_O)

# define loss function
loss_fn = torch.nn.MSELoss(reduction='sum')

# define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

for t in range(EPOCHS):
    # forward pass: compute predicted y
    y_p = model(x)

    # compute and print loss
    loss = loss_fn(y_p, y)
    print(t, loss.item())

    # backward pass
    optimizer.zero_grad()
    loss.backward()

    # update weights
    optimizer.step()
