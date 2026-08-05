import torch
import numpy as np
class Memory:

    def __init__(self, size, batch_size, data_shape):
        super().__init__()
        self.data = torch.zeros((size, batch_size, *data_shape))
        self.label = torch.zeros((size, batch_size))
        self.ids = torch.ones(size, dtype=torch.long)*-1
        self.index = 0
        self.size = size

    def save_batch(self, batch, batch_id):
        if self.index == self.size:
            raise MemoryError("MEMORY IS FULL")
        self.data[self.index] = batch[0]
        self.label[self.index] = batch[1]
        self.ids[self.index] = batch_id
        self.index +=1
        return

    def get_batches(self, ids):

        return list(zip(self.data[ids], self.label[ids]))

    def drop_batches(self, to_drop_ids):
        to_keep_ids = np.arange(self.index)[np.array([x not in to_drop_ids for x in range(self.index)])]

        to_ret = self.ids[to_drop_ids]
        # SHIFT CONCEPT TO KEEP

        self.data[:len(to_keep_ids)] = self.data[to_keep_ids]
        self.label[:len(to_keep_ids)] = self.label[to_keep_ids]
        self.ids[:len(to_keep_ids)] = self.ids[to_keep_ids]

        # CLEAR BUFFER TAIL
        self.index = len(to_keep_ids)
        self.data[self.index:] = 0
        self.label[self.index:] = 0
        self.ids[self.index:] = -1
        return to_ret

    def drop_all(self):
        self.data[:,:,:] = 0
        self.label[:,:] = 0
        self.ids[:] = -1
        self.index = 0
        return

    def drop_last(self):
        self.data[:-1] = self.data[1:]
        self.label[:-1] = self.label[1:]
        self.ids[:-1] = self.ids[1:]
        self.index -= 1
        return
