import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch


class DenseConditionalAutoEncoder(nn.Module):
    def __init__(self, input_size, latent_size, starting_epochs, decay_factor, device):
        super(DenseConditionalAutoEncoder, self).__init__()

        self.input_size = input_size
        self.epochs = starting_epochs
        self.device = device
        self.decay_factor = decay_factor
        self.encoder = nn.Linear(self.input_size, latent_size)
        self.decoder = nn.Linear(latent_size + 1, self.input_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.to(device)

    def forward(self, x, y):
        dims = x.shape
        x = x.flatten(start_dim=1)
        encoded = self.encoder(x)
        encoded = torch.cat((encoded, y.unsqueeze(1)), dim=1)
        return self.decoder(encoded).reshape(dims)

    def train_model(self, batches):
        self.epochs = max(4, int(self.epochs/self.decay_factor))
        self.train()  # Imposta il modello in modalità di addestramento

        criterion = nn.MSELoss()
        criterion_var = nn.MSELoss(reduction='none')

        for _ in range(self.epochs):
            for data, label in batches:
                data, label = data.to(self.device), label.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(data, label.float())
                loss = criterion(outputs, data)
                loss.backward()
                self.optimizer.step()

        with torch.no_grad():
            total_loss = 0
            size = 0
            squared_sum = 0
            for data, label in batches:
                data, label = data.to(self.device), label.to(self.device)
                outputs = self(data, label.float())
                loss = criterion(outputs, data)
                squared_sum += (criterion_var(outputs.detach(),  data).mean(dim=(1,2,3))**2).sum()
                total_loss += loss*data.shape[0]
                size += data.shape[0]

        return total_loss, size, squared_sum

    def get_error(self, data, label):
        with torch.no_grad():
            outputs = self(data, label.float())
            return nn.MSELoss()(outputs, data)



