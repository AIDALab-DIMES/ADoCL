from concept import AEConcept
from memory import Memory
import torch

class AEDetector:

    def __init__(self, trainer, num_batches, batch_size, data_shape, num_classes, theta, mem_size, eps, latent_dim, starting_epochs, decay_factor, device):

        # DATA
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.n_features = data_shape
        self.num_classes = num_classes

        # CONTINUOUS LERNER
        self.trainer = trainer
        self.device = device
        self.id_task  = -1

        # DETECTOR PARAMETERS
        self.theta = theta
        self.eps = eps
        self.mem_size = mem_size

        # CONCEPT STRUCTURE
        self.concept = None
        self.latent_dim = latent_dim
        self.starting_epochs = starting_epochs
        self.decay_factor = decay_factor

        # MEMORY STRUCTURE
        self.memory = Memory(mem_size, batch_size, data_shape)
        self.prediction = torch.ones((num_batches,2))*-1

        self.candidate_concepts = []

    def process(self, data, label, batch_id, verbose=False):

        log_buffer = []


        def log(*args, **kwargs):
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            log_buffer.append(sep.join(str(arg) for arg in args) + end)

        log("---- STATE ----")
        log("BATCH ID:", batch_id)
        log("CONCEPT THRES:", self.concept.get_thres(self.eps) if self.concept is not None else 0)
        log("MEMORY OCCUPANCY:",self.memory.index)
        log("NUMBER OF CANDIDATE:", len(self.candidate_concepts))
        max_candidate_size = 0
        s = ''
        for candidate in self.candidate_concepts:
            indexes = candidate.get_indexes()
            s += str(indexes) + '\n'
            max_candidate_size = max(len(indexes), max_candidate_size)
        log("MAX CANDIDATE SIZE:",max_candidate_size)
        log(s, end='')
        log("--- PROCESS ---")

        if self.concept is not None and self.concept.complies([(data, label)], self.eps):

            log("BATCH COMPLIES")
            self.concept.update([(data, label)])
            #self.trainer.train(batches = [(data, label)])
            self.prediction[batch_id,0] = 0
            self.prediction[batch_id,1] = self.id_task
            log(f"ERROR ON CURRENT CONCEPT: {self.concept.get_error((data, label)).item()}")
            log("UPDATING CONCEPT")

            to_drop_indexes = []
            while True:
                to_add_indexes = []
                to_drop_candidate = []
                for candidate in self.candidate_concepts:
                    batches = self.memory.get_batches(candidate.get_indexes())
                    if self.concept.complies(batches, self.eps):
                        to_add_indexes += candidate.get_indexes()
                        to_drop_candidate.append(candidate)
                if to_add_indexes:
                    log("UPDATING CONCEPT")
                    to_drop_indexes += to_add_indexes
                    for candidate in to_drop_candidate:
                        self.candidate_concepts.remove(candidate)
                    self.concept.update(self.memory.get_batches(to_add_indexes))
                    #self.trainer.train(batches = self.memory.get_batches(to_add_indexes))

                else:
                    break

            if to_drop_indexes:
                new_concept_batches = self.memory.drop_batches(to_drop_indexes)
                self.prediction[new_concept_batches, 1] = self.id_task
                for candidate in self.candidate_concepts:
                    candidate.fix_indexes(to_drop_indexes)

        else:

            log("BATCH DOESN'T COMPLY")
            self.memory.save_batch((data, label), batch_id)
            self.prediction[batch_id, 0] = 1

            min_error = float('+inf')
            min_candidate = None
            for candidate in self.candidate_concepts:
                if candidate.complies([(data, label)], self.eps):
                    error = candidate.get_error((data, label))
                    if error < min_error:
                        min_error = error
                        min_candidate = candidate


            if min_candidate is None:
                log("NEW CANDIDATE GENERATION")
                new_candidate = AEConcept(self.n_features.prod(), self.latent_dim, self.starting_epochs, self.decay_factor, self.device)
                new_candidate.update([(data, label)])
                new_candidate.update_indexes([self.memory.index-1])
                self.candidate_concepts.append(new_candidate)
                log(f"ERROR ON CURRENT CONCEPT: {self.concept.get_error((data, label)).item() if self.concept is not None else '+inf'}")
                log(f"ERROR NEW CANDIDATE: {new_candidate.get_error((data, label)).item()}")

                if self.memory.index == self.mem_size:
                    self.memory.drop_last()
                    for candidate in self.candidate_concepts:
                        if 0 in candidate.indexes:
                            candidate.indexes.remove(0)
                        candidate.fix_indexes(list(range(1, self.mem_size)))

                    for candidate in self.candidate_concepts:
                        if len(candidate.indexes) == 0:
                            self.candidate_concepts.remove(candidate)
                            break
            else:
                log(f"ERROR ON CURRENT CONCEPT: {self.concept.get_error((data, label)).item() if self.concept is not None else '+inf'}")
                log(f"THRES OF OLD CANDIDATE: {min_candidate.get_thres(self.eps)}")
                log(f"ERROR ON OLD CANDIDATE: {min_candidate.get_error((data, label)).item()}")
                log("UPDATING CONCEPT")
                self.candidate_concepts.remove(min_candidate)

                min_candidate.update([(data, label)])
                min_candidate.update_indexes([self.memory.index - 1])
                to_drop_indexes = []
                while True:
                    to_add_indexes = []
                    to_drop_candidate = []
                    for candidate in self.candidate_concepts:
                        batches = self.memory.get_batches(candidate.get_indexes())
                        if min_candidate.complies(batches, self.eps):
                            to_add_indexes += candidate.get_indexes()
                            to_drop_candidate.append(candidate)
                    if to_add_indexes:
                        log("UPDATING CONCEPT")
                        to_drop_indexes += to_add_indexes
                        for candidate in to_drop_candidate:
                            self.candidate_concepts.remove(candidate)
                        min_candidate.update(self.memory.get_batches(to_add_indexes))
                        min_candidate.update_indexes(to_add_indexes)
                    else:
                        break


                if len(min_candidate.get_indexes()) >= self.theta:
                    log("NEW CONCEPT FOUND")

                    self.concept = min_candidate
                    #self.trainer.train(batches = self.memory.get_batches(self.concept.get_indexes()))
                    new_concept_batches = self.memory.drop_batches(self.concept.get_indexes())
                    self.id_task +=1
                    self.prediction[new_concept_batches,1] = self.id_task
                    for candidate in self.candidate_concepts:
                        candidate.fix_indexes(self.concept.get_indexes())
                else:
                    self.candidate_concepts.append(min_candidate)
                    if self.memory.index == self.mem_size:
                        self.memory.drop_last()
                        for candidate in self.candidate_concepts:
                            if 0 in candidate.indexes:
                                candidate.indexes.remove(0)
                            candidate.fix_indexes(list(range(1, self.mem_size)))

                        for candidate in self.candidate_concepts:
                            if len(candidate.indexes) == 0:
                                self.candidate_concepts.remove(candidate)
                                break

        log("---------------\n")

        if verbose:
            print("".join(log_buffer))
        return
