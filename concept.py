from models.model4concept import DenseConditionalAutoEncoder


class AEConcept:

    def __init__(self, input_size, latent_dim, staring_epochs, decay_factor, device, sigma=1.5):
        super().__init__()

        self.model = DenseConditionalAutoEncoder(input_size, latent_dim, staring_epochs, decay_factor, device)
        self.model.to(device)
        self.device = device

        self.indexes = []
        self.error = 0
        self.squared_error = 0
        self.size = 0
        self.sigma = sigma

    def complies(self, batches, eps):

        if self.size == 0:
            return True

        mean = self.error / self.size
        variance = (self.squared_error / self.size) - (mean ** 2)
        std_dev = variance ** 0.5

        for data, label in batches:
            data, label = (data.to(self.device), label.to(self.device))
            error = self.model.get_error(data, label.float())
            if error > mean + self.sigma*std_dev:
                return False

        return True

    def get_error(self, batch):
        data, label = (batch[0].to(self.device), batch[1].to(self.device))
        return self.model.get_error(data, label.float())


    def update(self, batches):
        loss, size, squared = self.model.train_model(batches)
        self.error = loss
        self.size = size
        self.squared_error = squared
        return

    def get_indexes(self):
        return self.indexes

    def update_indexes(self, new_indexes):
        self.indexes += new_indexes
        return

    def fix_indexes(self, dropped_indexes):
        for i in range(len(self.indexes)):
            self.indexes[i] -= sum(1 for x in dropped_indexes if x < self.indexes[i])
        return

    def get_thres(self, eps):
        if self.size == 0:
            return None

        mean = self.error / self.size
        variance = (self.squared_error / self.size) - (mean ** 2)
        std_dev = variance ** 0.5
        return (mean + self.sigma*std_dev).item()