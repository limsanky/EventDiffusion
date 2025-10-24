import random
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler, BatchSampler
import os
from PIL import Image
from mvsec_helper import SEQUENCES_FRAMES

class SingleMVSECSampler(Sampler):
    def __init__(self, scenario: str, split: int):
        self.scenario = scenario
        self.split = split
        assert split in [1, 2, 3]
        if split == 1:
            self.training_splits = [2, 3]
        elif split == 2:
            self.training_splits = [1, 3]
        elif split == 3:
            self.training_splits = [1, 2]
        
        count = {}
        self.indices = {}
        for exp in self.training_splits:
            first_index, last_index = SEQUENCES_FRAMES[self.scenario][f'split{self.split}'][self.scenario + str(exp)]
            self.indices[exp] = (first_index, last_index)
            count[exp] = last_index - first_index + 1
            count[exp] = count[exp] - 1  # Because we return a PAIR of images
            # print(f'Number of samples for {self.scenario}{exp} in split {self.split}: {count[exp]}')

        self.count = count

    # def __iter__(self):
    #     indices = []
    #     for exp in self.training_splits:
    #         idxs = list(range(1, self.count[exp] + 1))
    #         random.shuffle(idxs)
    #         indices.append(idxs)

    #     for first, second in zip(*indices):
    #         yield [ first, second ]
    
    def __iter__(self):
        indices = []
        
        for exp in self.training_splits:
            # idxs = list(range(self.indices[exp][0] + 1, self.indices[exp][1] + 1))
            # assert len(idxs) == self.count[exp], f"Length mismatch for {self.scenario}{exp} in split {self.split}: expected {self.count[exp]}, got {len(idxs)}"
            # for _i in range(len(idxs)):
            #     assert idxs[_i] > self.indices[exp][0], f"Index out of bounds: {idxs[_i]} is smaller than or equal to {self.indices[exp][0]}"
            #     assert idxs[_i] <= self.indices[exp][1], f"Index out of bounds: {idxs[_i]} is greater than {self.indices[exp][1]}"
            idxs = list(range(1, self.count[exp] + 1))
            random.shuffle(idxs)
            indices.append(idxs)

        for first, second in zip(*indices):
            yield [ first, second ]
    
    def __len__(self):
        output = sum(self.count.values())
        output = 2 * output # Since there are left AND right cameras
        return output

class MVSECSampler(BatchSampler):
    def __init__(self, sampler, batch_size: int, drop_last: bool = False):
        super().__init__(sampler, batch_size, drop_last)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for indices in self.sampler:
            # print('indices', indices)
            # exit()
            batch.append(indices)
            if len(batch) == self.batch_size:
                # print('yay')
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # print('hi', sum(self.sampler.count.values()))
        # exit()
        total_count = self.sampler.__len__()
        # print('total_count', total_count, sum(self.sampler.count.values()))
        # exit()
        
        if self.drop_last:
            return total_count // self.batch_size
        
        length = (total_count + self.batch_size - 1) // self.batch_size
        # print('length', length)
        # exit()
        return length

class MVSECDataset(Dataset):

    def __init__(self, data_dir: str, scenario: str, split: int, event_transforms=None, image_transforms=None):
        super().__init__()
        self.loc = ['left', 'right']
        self.data_dir = data_dir
        self.scenario = scenario
        self.event_transforms = event_transforms
        self.image_transforms = image_transforms
        self.split = split
        assert split in [1, 2, 3]
        if split == 1:
            self.training_splits = (2, 3)
        elif split == 2:
            self.training_splits = (1, 3)
        elif split == 3:
            self.training_splits = (1, 2)

        self.event_data = self.load_event_data()
        self.image_paths = self.load_image_paths()
        self.len_of_events = len(self.event_data[self.loc[0]][self.training_splits[0]]) + len(self.event_data[self.loc[0]][self.training_splits[1]])
        self.len_of_events += len(self.event_data[self.loc[1]][self.training_splits[0]]) + len(self.event_data[self.loc[1]][self.training_splits[1]])
        self.len_of_images = len(self.image_paths[self.loc[0]][self.training_splits[0]]) + len(self.image_paths[self.loc[0]][self.training_splits[1]])
        self.len_of_images += len(self.image_paths[self.loc[1]][self.training_splits[0]]) + len(self.image_paths[self.loc[1]][self.training_splits[1]])
        assert self.len_of_events == self.len_of_images, f"Mismatch between event {self.len_of_events} and image data lengths {self.len_of_images}"

    def load_event_data(self) -> torch.Tensor:
        event_data = {}
        for loc in self.loc:
            data = {}
            for exp in self.training_splits:
                ev_rep_path = os.path.join(self.data_dir, f'{self.scenario}/{self.scenario}{exp}/evrep_train/evrep_{loc}.npy')
                # ev_rep_path = os.path.join(self.data_dir, f'{self.scenario}/{self.scenario}{exp}/evrep_train/evrep_{loc}.pt')
                assert os.path.exists(ev_rep_path), f'EvRep file does not exist: {ev_rep_path}'
                # print(f'Loading EvRep for {self.scenario}{exp} at {loc} camera.')
                ev_rep = np.load(ev_rep_path)
                data[exp] = torch.from_numpy(ev_rep)
                # data[exp] = torch.load(ev_rep_path)
                
            event_data[loc] = data
            
        return event_data
    
    def load_image_paths(self) -> dict:
        image_paths = {}
        for loc in self.loc:
            image_paths_for_split = {}
            for exp in self.training_splits:
                individual_image_paths = []
                first_index, last_index = SEQUENCES_FRAMES[self.scenario][f'split{self.split}'][self.scenario + str(exp)]
                image_path = os.path.join(self.data_dir, f'{self.scenario}/{self.scenario}{exp}/images/{loc}/')
                assert os.path.exists(image_path), f'Image directory does not exist: {image_path}'
                for idx, img in enumerate(os.listdir(image_path)):
                    if idx < first_index or idx > last_index:
                        continue
                    individual_image_path = os.path.join(image_path, img)
                    individual_image_paths.append(individual_image_path)
                    
                image_paths_for_split[exp] = individual_image_paths
            
            image_paths[loc] = image_paths_for_split
        
        return image_paths
    
    def __len__(self):
        return len(self.event_data) - 1
    
    def __getitem__(self, index):
        image_data = {}
        event_data = {}
        
        for loc in self.loc:
            
            loc_event_data: dict = self.event_data[loc]
            
            event_data_at_loc = [ (loc_event_data.get(s)[i], i) for s, i in zip(self.training_splits, index) ]
            
            if self.event_transforms:
                event_data_at_loc = [ (self.event_transforms(_events), _i) for (_events, _i) in event_data_at_loc ]
            
            loc_image_data_at_split = [ self.image_paths[loc][s] for s in self.training_splits ]
            images_at_t0 = [ (Image.open(loc_image_data_at_split[pos][i - 1]), i - 1) for pos, i in enumerate(index) ]
            images_at_t1 = [ (Image.open(loc_image_data_at_split[pos][i]), i) for pos, i in enumerate(index) ]

            if self.image_transforms:
                images_at_t0 = [ (self.image_transforms(img), i) for (img, i) in images_at_t0 ]
                images_at_t1 = [ (self.image_transforms(img), i) for (img, i) in images_at_t1 ]

            event_data[loc] = event_data_at_loc
            image_data[loc] = { f'{s}': (images_at_t0[0], images_at_t1[0]) for s in self.training_splits }
            
        return (event_data, image_data)
    