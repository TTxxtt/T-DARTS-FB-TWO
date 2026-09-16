import torch
import random
import numpy as np
from itertools import combinations
from torch.utils.data import DataLoader

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        
def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def random_choice(m):
    assert m >= 1
    
    choice = {}
    # m_ = np.random.randint(low=1, high=m+1, size=1)[0]
    # choice_list = random.sample(range(m), m_)
    # choice['op'] = choice_list 
    m_low = np.random.randint(low=1, high=m+1, size=1)[0]
    low_list = random.sample(range(4), m_low)

    m_mid = np.random.randint(low=1, high=m+1, size=1)[0]
    mid_list = random.sample(range(4), m_mid)

    m_high = np.random.randint(low=1, high=m+1, size=1)[0]
    high_list = random.sample(range(4), m_high)
   
    # # ops = []
    # # for i in range(m_):
    # #     ops.append(random.sample(range(4), 1)[0])
    #     # ops.append(random.sample(range(2), 1)[0])

    # # choice['op'] = ops
    choice['Low'] = low_list
    choice['Mid'] = mid_list
    choice['High'] = high_list
    
    return choice

def find_choice_index(m, choice): #升序排列
    choice_list = []
    ops = [0, 1, 2, 3]
    for m_ in range(1, m+1):
        choices = combinations(ops, m_)
        for id, operate in enumerate(choices):           
            choice_list.append(list(operate))
    choice = sorted(choice) 
    index = choice_list.index(choice)
    return index
    
    

def conv_2_matrix(choice):
    op_ids = choice['op']
    path_ids = choice['path']
    selections = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    
    ops = ['input']
    for i in range(4):  # 初始默认操作
        ops.append(selections[0])
    for i, id in enumerate(path_ids):  # 按choice修改
        ops[id + 1] = selections[op_ids[i]]
    ops.append('conv1x1-bn-relu')
    ops.append('output')
    
    matrix = np.zeros((7, 7), dtype=np.int)
    for id in path_ids:
        matrix[0, id + 1] = 1
        matrix[id + 1, 5] = 1
    matrix[5, -1] = 1
    matrix = matrix.tolist()
    model_spec = api.ModelSpec(matrix=matrix, ops=ops)
    
    return model_spec


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


def set_seed(seed):
    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def sort_list_with_index(input_list):
    sorted_list = sorted(zip(range(len(input_list)), input_list), key=lambda x: x[1])
    return [x[0] for x in sorted_list]

def traverse_choice(m):
    choice_list = []
    ops = [0, 1, 2, 3]
    for m_ in range(1, m+1):
        choices = combinations(ops, m_)
        for id, operate in enumerate(choices):           
            choice_list.append(list(operate))
    choice_dict_list = []
    for n1 in range(len(choice_list)):
        for n2 in range(len(choice_list)):
            for n3 in range(len(choice_list)):
                choice_dict_list.append({'Low':choice_list[n1],'Mid':choice_list[n2],'High':choice_list[n3]})
    
    # print(len(choice_dict_list))
    return choice_dict_list


class CutMix:
    def __init__(self, probability=[0.9,0.7,0.5], min_area=0.02, max_area=1/3, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        self.device = device

    def _cutmix(self, x1, x2, width):
        if random.random() > self.probability[0]:
            target_area = random.uniform(self.min_area, self.max_area) * width
            w = int(target_area)
            left = random.randint(0, width - w)
            x1[:,left:left + w,:,0:3] = x2[:,left:left + w,:,0:3]
        if random.random() > self.probability[1]:
            target_area = random.uniform(self.min_area, self.max_area) * width
            w = int(target_area)
            left = random.randint(0, width - w)
            x1[:,left:left + w,:,3:6] = x2[:,left:left + w,:,3:6]
        if random.random() > self.probability[2]:
            target_area = random.uniform(self.min_area, self.max_area) * width
            w = int(target_area)
            left = random.randint(0, width - w)
            x1[:,left:left + w,:,6:9] = x2[:,left:left + w,:,6:9]        
            
    def __call__(self, input, labels):
        batch_size, _, width, chan, filter = input.size()
        labels = labels.cpu()
        types = torch.unique(labels).tolist()
        for c in types:
            idxes = torch.where(labels == c)[0].tolist()
            for i in range(len(idxes)-1):
                self._cutmix(input[idxes[i]], input[idxes[i+1]], width)
        return input

class RandomErasing:
    """
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased time duration wrt input EEG trials.
         max_area: Maximum percentage of erased time duration wrt input EEG trials.
    """

    def __init__(self, probability=[0.9,0.7,0.5], min_area=0.02, max_area=1/3, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        self.device = device

    def _erase(self, x, chan, width):
        if random.random() > self.probability[0]:
            target_area = random.uniform(self.min_area, self.max_area) * width
            w = int(target_area)
            left = random.randint(0, width - w)
            x[:,left:left + w,:,0:3].zero_()
        if random.random() > self.probability[1]:
            target_area = random.uniform(self.min_area, self.max_area) * width
            w = int(target_area)
            left = random.randint(0, width - w)
            x[:,left:left + w,:,3:6].zero_()
        if random.random() > self.probability[2]:
            target_area = random.uniform(self.min_area, self.max_area) * width
            w = int(target_area)
            left = random.randint(0, width - w)
            x[:,left:left + w,:,6:9].zero_()

    def __call__(self, input, *args):
        if len(input.size()) == 2:
            self._erase(input, *input.size())
        else:
            batch_size, _, width, chan, filter = input.size()
            for i in range(batch_size):
                self._erase(input[i], chan, width)
        return input
    
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, *args):
        for t in self.transforms:
            x = t(x, *args)
        return x

def build_tranforms():
    return Compose([
        CutMix(),  #used in IFNet V2
        RandomErasing(),
    ])


class RepeatedTrialAugmentation:
    def __init__(self, transform = torch.nn.Identity(), m = 5):
        self.transform = transform
        self.m = m

    def __call__(self, x, y):
        if self.m == 1:
            return self.transform(x, y), y

        # todo: could be run in parallel
        # bdata = Parallel(n_jobs=self.m)(delayed(self.transform)(x, y) for _ in range(self.m))
        # data = torch.cat(bdata, dim=0)

        # Need x.clone() if transforms are in-place!
        data = torch.cat([self.transform(x.clone(), y) for _ in range(self.m)], dim=0)
        
        # data = torch.cat((x, data), dim=0)
        
        labels = torch.cat([y]*self.m, dim=0)
        return data, labels

 
    
    

if __name__ == '__main__':
    set_seed(2020)
    for i in range(10):
        choice = random_choice(m=2)
        print(choice)
    # choice_list = traverse_choice(m=2)
    # print(choice_list[0]['Low'])

