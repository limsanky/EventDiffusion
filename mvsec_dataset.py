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

    def __iter__(self):
        indices = []
        for exp in self.training_splits:
            idxs = list(range(1, self.count[exp] + 1))
            random.shuffle(idxs)
            indices.append(idxs)

        for first, second in zip(*indices):
            yield [first, second]
    
    def __len__(self):
        return self.count

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
        total_count = sum(self.sampler.count.values())
        if self.drop_last:
            return total_count // self.batch_size
        else:
            return (total_count + self.batch_size - 1) // self.batch_size

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
            self.training_splits = [2, 3]
        elif split == 2:
            self.training_splits = [1, 3]
        elif split == 3:
            self.training_splits = [1, 2]

        self.event_data = self.load_event_data()
        self.image_paths = self.load_image_paths()
        len_of_events = len(self.event_data[self.loc[0]][0]) + len(self.event_data[self.loc[0]][1])
        len_of_events += len(self.event_data[self.loc[1]][0]) + len(self.event_data[self.loc[1]][1])
        len_of_images = len(self.image_paths[self.loc[0]]) + len(self.image_paths[self.loc[1]])
        assert len_of_events == len_of_images, f"Mismatch between event {len_of_events} and image data lengths {len_of_images}"

    def load_event_data(self) -> torch.Tensor:
        event_data = {}
        for loc in self.loc:
            data = []
            for exp in self.training_splits:
                ev_rep_path = os.path.join(self.data_dir, f'{self.scenario}/{self.scenario}{exp}/evrep_train/evrep_{loc}.npy')
                assert os.path.exists(ev_rep_path), f'EvRep file does not exist: {ev_rep_path}'
                ev_rep = np.load(ev_rep_path)
                # print(ev_rep.shape)
                data.append(torch.from_numpy(ev_rep))
            # print(data[0].shape[0] + data[1].shape[0])
            event_data[loc] = data
        return event_data
    
    def load_image_paths(self) -> dict:
        image_paths = {}
        for loc in self.loc:
            individual_image_paths = []
            for exp in self.training_splits:
                first_index, last_index = SEQUENCES_FRAMES[self.scenario][f'split{self.split}'][self.scenario + str(exp)]
                image_path = os.path.join(self.data_dir, f'{self.scenario}/{self.scenario}{exp}/images/{loc}/')
                assert os.path.exists(image_path), f'Image directory does not exist: {image_path}'
                for idx, img in enumerate(os.listdir(image_path)):
                    if idx < first_index or idx > last_index:
                        continue
                    individual_image_path = os.path.join(image_path, img)
                    individual_image_paths.append(individual_image_path)
            # print(len(individual_image_paths))
            image_paths[loc] = individual_image_paths
        # exit()
        return image_paths
    
    def __len__(self):
        return len(self.event_data) - 1
    
    def __getitem__(self, index):
        # print('hi', index)
        # exit()
    
        image_data = {}
        event_data = {}
        
        for loc in self.loc:
            
            loc_event_data = self.event_data[loc]
            # print(loc_event_data[0].shape, loc_event_data[1].shape)
            # exit()
            event_data_at_loc = [ loc_event_data[i][index[i] + 1] for i in range(len(self.training_splits)) ]
            
            if self.event_transforms:
                event_data_at_loc = self.event_transforms(event_data_at_loc)
                    
            for i in range(len(self.training_splits)):
                images_at_t0 = Image.open(self.image_paths[loc][index[i]])
                images_at_t1 = Image.open(self.image_paths[loc][index[i] + 1])
                
                if self.image_transforms:
                    images_at_t0 = self.image_transforms(images_at_t0)
                    images_at_t1 = self.image_transforms(images_at_t1)
                    
            event_data[loc] = event_data_at_loc
            image_data[loc] = (images_at_t0, images_at_t1)
            
        return (event_data, image_data)
    