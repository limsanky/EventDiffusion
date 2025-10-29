import random
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler, BatchSampler
import os
from PIL import Image
from mvsec_helper import SEQUENCES_FRAMES

class SingleMVSECSampler(Sampler):
    def __init__(self, scenario: str, split: int, is_training: bool):
        self.scenario = scenario
        self.split = split
        assert split in [1, 2, 3]
        if split == 1:
            self.training_splits = [2, 3]
        elif split == 2:
            self.training_splits = [1, 3]
        elif split == 3:
            self.training_splits = [1, 2]
        self.is_training = is_training
        count = {}
        self.indices = {}
        
        if is_training:
            for exp in self.training_splits:
                first_index, last_index = SEQUENCES_FRAMES[self.scenario][f'split{self.split}'][self.scenario + str(exp)]
                self.indices[exp] = (first_index, last_index)
                count[exp] = last_index - first_index + 1
                count[exp] = count[exp] - 1  # Because we return a PAIR of images
                # print(f'Number of samples for {self.scenario}{exp} in split {self.split}: {count[exp]}')
        else:
            exp = self.split
            first_index, last_index = SEQUENCES_FRAMES[self.scenario][f'split{self.split}'][self.scenario + str(exp)]
            self.indices[exp] = (first_index, last_index)
            count[exp] = last_index - first_index + 1
            count[exp] = count[exp] - 1  # Because we return a PAIR of images
        
        self.count = count
    
    def __iter__(self):
        # indices = []
        
        # for exp in self.training_splits:
        #     idxs = list(range(1, self.count[exp] + 1))
        #     # random.shuffle(idxs)
        #     indices.append(idxs)

        # for first, second in zip(*indices):
        #     yield [ first, second ]
        
        return_indices = []
        all_indices = np.random.randint(1, self.__len__() + 1, size=(self.__len__(),))
        # all_indices = [ i for i in range(1, self.__len__() + 1) ]
        # all_indices.reverse()
        # all_indices = np.random.randint(1415, 1425, size=(10,))
        total_count = 0

        if self.is_training:
            for exp in self.training_splits:
                split_count = self.count[exp]
                for i in all_indices:
                    if (i <= total_count + split_count) and (i > total_count):
                        idx = i - total_count
                        return_indices.append((idx, exp))
                        # if exp == 3:
                        #     print(f'appended index {idx} for exp {exp}')
                        #     exit()
                        # else:
                        #     print(f'appended index {idx} for exp {exp}')
                total_count += split_count
        else:
            exp = self.split
            split_count = self.count[exp]
            for i in all_indices:
                if (i <= total_count + split_count) and (i > total_count):
                    idx = i - total_count
                    return_indices.append((idx, exp))
            total_count += split_count
        
        assert len(return_indices) == self.__len__(), f'Length mismatch: {len(return_indices)} vs {self.__len__()}'

        for idx, exp in return_indices:
            yield (idx, exp)
        
    
    def __len__(self):
        output = sum(self.count.values())
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

    def __init__(self, data_dir: str, scenario: str, split: int, is_training: bool, event_transforms=None, image_transforms=None, get_depth: bool = False, depth_transforms=None, get_disparity: bool = False, disparity_transforms=None):
        super().__init__()
        self.loc = ['left', 'right']
        self.data_dir = data_dir
        self.scenario = scenario
        self.is_training = is_training
        self.event_transforms = event_transforms
        self.image_transforms = image_transforms
        self.depth_transforms = depth_transforms
        self.disparity_transforms = disparity_transforms
        
        self.get_depth = get_depth
        self.get_disparity = get_disparity
        if get_depth:
            assert not get_disparity, "Cannot get both depth and disparity data."
            assert disparity_transforms is None
        if get_disparity:
            assert not get_depth, "Cannot get both depth and disparity data."
            assert depth_transforms is None
        
        self.split = split
        assert split in [1, 2, 3]
        if split == 1:
            self.training_splits = (2, 3)
        elif split == 2:
            self.training_splits = (1, 3)
        elif split == 3:
            self.training_splits = (1, 2)

        if is_training:
            self.event_data = self.load_event_data()
            self.image_paths = self.load_image_paths()
            self.len_of_events = len(self.event_data[self.loc[0]][self.training_splits[0]]) + len(self.event_data[self.loc[0]][self.training_splits[1]])
            self.len_of_events += len(self.event_data[self.loc[1]][self.training_splits[0]]) + len(self.event_data[self.loc[1]][self.training_splits[1]])
            self.len_of_images = len(self.image_paths[self.loc[0]][self.training_splits[0]]) + len(self.image_paths[self.loc[0]][self.training_splits[1]])
            self.len_of_images += len(self.image_paths[self.loc[1]][self.training_splits[0]]) + len(self.image_paths[self.loc[1]][self.training_splits[1]])
            
            if get_depth:
                self.depth_data_paths = self.load_depth_data()
                self.len_of_depths = len(self.depth_data_paths[self.training_splits[0]]) + len(self.depth_data_paths[self.training_splits[1]])
                assert (2 * self.len_of_depths) == self.len_of_images, f"Mismatch between 2x depth {2 * self.len_of_depths}, and image data lengths {self.len_of_images}"
            else:
                self.depth_data_paths = None
                self.len_of_depths = 0
            
            if get_disparity:
                self.disparity_data_paths = self.load_disparity_data()
                self.len_of_disparities = len(self.disparity_data_paths[self.training_splits[0]]) + len(self.disparity_data_paths[self.training_splits[1]])
                assert (2 * self.len_of_disparities) == self.len_of_images, f"Mismatch between 2x disparity {2 * self.len_of_disparities}, and image data lengths {self.len_of_images}"
            else:
                self.disparity_data_paths = None
                self.len_of_disparities = 0
        else:
            self.event_data = self.load_event_test_data()
            self.image_paths = self.load_image_test_paths()
            self.len_of_events = len(self.event_data[self.loc[0]][self.split]) 
            self.len_of_events += len(self.event_data[self.loc[1]][self.split])
            self.len_of_images = len(self.image_paths[self.loc[0]][self.split])
            self.len_of_images += len(self.image_paths[self.loc[1]][self.split])
            
            if get_depth:
                self.depth_data_paths = self.load_depth_test_data()
                self.len_of_depths = len(self.depth_data_paths[self.split])
                assert (2 * self.len_of_depths) == self.len_of_images, f"Mismatch between 2x depth {2 * self.len_of_depths}, and image data lengths {self.len_of_images}"
            else:
                self.depth_data_paths = None
                self.len_of_depths = 0
            
            if get_disparity:
                self.disparity_data_paths = self.load_disparity_test_data()
                self.len_of_disparities = len(self.disparity_data_paths[self.split])
                assert (2 * self.len_of_disparities) == self.len_of_images, f"Mismatch between 2x disparity {2 * self.len_of_disparities}, and image data lengths {self.len_of_images}"
            else:
                self.disparity_data_paths = None
                self.len_of_disparities = 0

        assert self.len_of_events == self.len_of_images, f"Mismatch between event {self.len_of_events} and image data lengths {self.len_of_images}"
    
    def load_depth_data(self):
        depth_paths = {}
        for exp in self.training_splits:
            individual_depth_paths = []
            first_index, last_index = SEQUENCES_FRAMES[self.scenario][f'split{self.split}'][self.scenario + str(exp)]
            depth_path = os.path.join(self.data_dir, f'{self.scenario}/{self.scenario}{exp}/depth_gt/')
            assert os.path.exists(depth_path), f'Depth directory does not exist: {depth_path}'
            for idx, img in enumerate(os.listdir(depth_path)):
                if idx < first_index or idx > last_index:
                    continue
                individual_depth_path = os.path.join(depth_path, img)
                individual_depth_paths.append(individual_depth_path)
                
            depth_paths[exp] = individual_depth_paths
        #     print(exp, len(individual_depth_paths), individual_depth_paths[0], individual_depth_paths[-1])
        # exit()
        return depth_paths

    def load_disparity_data(self):
        disparity_paths = {}
        for exp in self.training_splits:
            individual_disparity_paths = []
            first_index, last_index = SEQUENCES_FRAMES[self.scenario][f'split{self.split}'][self.scenario + str(exp)]
            disparity_path = os.path.join(self.data_dir, f'{self.scenario}/{self.scenario}{exp}/disparity_gt/')
            assert os.path.exists(disparity_path), f'Disparity directory does not exist: {disparity_path}'
            for idx, img in enumerate(os.listdir(disparity_path)):
                if idx < first_index or idx > last_index:
                    continue
                individual_disparity_path = os.path.join(disparity_path, img)
                individual_disparity_paths.append(individual_disparity_path)

            disparity_paths[exp] = individual_disparity_paths
        return disparity_paths

    def load_depth_test_data(self):
        depth_paths = {}
        exp = self.split
        individual_depth_paths = []
        first_index, last_index = SEQUENCES_FRAMES[self.scenario][f'split{self.split}'][self.scenario + str(exp)]
        depth_path = os.path.join(self.data_dir, f'{self.scenario}/{self.scenario}{exp}/depth_gt/')
        assert os.path.exists(depth_path), f'Depth directory does not exist: {depth_path}'
        for idx, img in enumerate(os.listdir(depth_path)):
            if idx < first_index or idx > last_index:
                continue
            individual_depth_path = os.path.join(depth_path, img)
            individual_depth_paths.append(individual_depth_path)
            
        depth_paths[exp] = individual_depth_paths
        return depth_paths
    
    def load_event_test_data(self):
        event_data = {}
        exp = self.split
        for loc in self.loc:
            data = {}
            ev_rep_path = os.path.join(self.data_dir, f'{self.scenario}/{self.scenario}{exp}/evrep_test/evrep_{loc}.pt')
            assert os.path.exists(ev_rep_path), f'EvRep file does not exist: {ev_rep_path}'
            print(f'Loading EvRep for {self.scenario}{exp} at {loc} camera.')
            ev_rep = torch.load(ev_rep_path)
            data[exp] = ev_rep
            
            event_data[loc] = data
            
        return event_data
    
    def load_image_test_paths(self) -> dict:
        image_paths = {}
        exp = self.split
        for loc in self.loc:
            image_paths_for_split = {}
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
    
    def load_event_data(self) -> torch.Tensor:
        event_data = {}
        for loc in self.loc:
            data = {}
            for exp in self.training_splits:
                # ev_rep_path = os.path.join(self.data_dir, f'{self.scenario}/{self.scenario}{exp}/evrep_train/evrep_{loc}.npy')
                ev_rep_path = os.path.join(self.data_dir, f'{self.scenario}/{self.scenario}{exp}/evrep_train/evrep_{loc}.pt')
                assert os.path.exists(ev_rep_path), f'EvRep file does not exist: {ev_rep_path}'
                # print(f'Loading EvRep for {self.scenario}{exp} at {loc} camera.')
                # ev_rep = np.load(ev_rep_path)
                # data[exp] = torch.from_numpy(ev_rep)
                data[exp] = torch.load(ev_rep_path)
                
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
        
        idx, split = index
        assert idx - 1 >= 0, f'Index must be at least 1 to get image pair, got {idx}'
        
        for loc in self.loc:
            
            loc_event_data: dict = self.event_data[loc]
            
            event_data_at_loc =  loc_event_data.get(split)
            # print('idx, split:', idx, split)
            # print('event_data_at_loc', event_data_at_loc.shape)
            event_data_at_loc = event_data_at_loc[idx]
            # print('event_data_at_loc[idx]', event_data_at_loc.shape)
            # exit()
            
            if self.event_transforms:
                event_data_at_loc = self.event_transforms(event_data_at_loc)
            
            loc_image_data_at_split = self.image_paths[loc][split]
            # print('split and index', split, idx)
            # print('image path for t0:', loc_image_data_at_split[idx - 1])
            # exit()
            # print(loc, loc_image_data_at_split[idx - 1])
            image_at_t0 = Image.open(loc_image_data_at_split[idx - 1])
            image_at_t1 = Image.open(loc_image_data_at_split[idx])
            # if loc == 'left':
            #     print('image left t0:', loc_image_data_at_split[idx - 1])
            #     print('image left t1:', loc_image_data_at_split[idx])
            # images_at_t0 = [ (Image.open(loc_image_data_at_split[pos][i - 1]), i - 1) for pos, i in enumerate(index) ]
            # images_at_t1 = [ (Image.open(loc_image_data_at_split[pos][i]), i) for pos, i in enumerate(index) ]

            if self.image_transforms:
                image_at_t0 = self.image_transforms(image_at_t0)
                image_at_t1 = self.image_transforms(image_at_t1)
            
            event_data[loc] = event_data_at_loc
            image_data[loc] = (image_at_t0, image_at_t1)
        
        depth_data = -1
        if self.get_depth:
            depth_path = self.depth_data_paths[split][idx]
            
            depth_data = Image.open(depth_path)
            if self.depth_transforms is not None:
                depth_data = self.depth_transforms(depth_data)

        disparity_data = -1
        if self.get_disparity:
            disparity_path = self.disparity_data_paths[split][idx]
            
            disparity_data = Image.open(disparity_path)
            if self.disparity_transforms is not None:
                disparity_data = self.disparity_transforms(disparity_data)

        return (event_data, image_data, depth_data, disparity_data)

    # def __getitem__(self, index):
    #     image_data = {}
    #     event_data = {}
        
    #     for loc in self.loc:
            
    #         loc_event_data: dict = self.event_data[loc]
            
    #         event_data_at_loc = [ (loc_event_data.get(s)[i], i) for s, i in zip(self.training_splits, index) ]
            
    #         if self.event_transforms:
    #             event_data_at_loc = [ (self.event_transforms(_events), _i) for (_events, _i) in event_data_at_loc ]
            
    #         loc_image_data_at_split = [ self.image_paths[loc][s] for s in self.training_splits ]
    #         images_at_t0 = [ (Image.open(loc_image_data_at_split[pos][i - 1]), i - 1) for pos, i in enumerate(index) ]
    #         images_at_t1 = [ (Image.open(loc_image_data_at_split[pos][i]), i) for pos, i in enumerate(index) ]

    #         if self.image_transforms:
    #             images_at_t0 = [ (self.image_transforms(img), i) for (img, i) in images_at_t0 ]
    #             images_at_t1 = [ (self.image_transforms(img), i) for (img, i) in images_at_t1 ]

    #         event_data[loc] = event_data_at_loc
    #         image_data[loc] = { f'{s}': (images_at_t0[0], images_at_t1[0]) for s in self.training_splits }
            
    #     return (event_data, image_data)
    