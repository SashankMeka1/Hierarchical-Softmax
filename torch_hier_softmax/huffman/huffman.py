import torch
import torch.nn as nn

class Node:
    def __init__(self,freq,word,rep_vector,left,right):
        self.FREQ = freq
        self.REP_VECTOR = rep_vector
        self.left = left
        self.right = right
        self.WORD = word

class HuffmanTree(nn.Module):
    def __init__(self,vocab_dict,rep_vector_size, device):
        super(HuffmanTree, self).__init__()
        self.ROOT = None
        self.PATHS = {}
        self.PARAMS = []
        self.DEVICE = device
        self.REP_VECTOR_SIZE = rep_vector_size
        self.VOCAB_DICT = vocab_dict
        self.VOCAB = []
        for word,freq in list(vocab_dict.items()):#build heap of nodes
            self.VOCAB.append(Node(freq,word,None,None,None))
        self.build_huffman_tree()
        self.assign_codes(self.ROOT,[])#assign codes
        self.PARAMS = nn.ParameterList(self.PARAMS)
    def heapify(self,idx):
        minimum = idx
        left = (2*idx)+1
        right = (2*idx)+2

        #recursive heapify smaller frequency nodes first with higher freq as children
        if(left < len(self.VOCAB) and self.VOCAB[left].FREQ < self.VOCAB[minimum].FREQ):
            minimum = left
        if(right < len(self.VOCAB) and self.VOCAB[right].FREQ < self.VOCAB[minimum].FREQ):
            minimum = right
        if minimum != idx:
            self.VOCAB[minimum], self.VOCAB[idx] = self.VOCAB[idx], self.VOCAB[minimum]
            self.heapify(minimum)
    def build_heap(self):
        for idx in range(len(self.VOCAB)//2):
            self.heapify(idx)
    def build_huffman_tree(self):
        self.build_heap()
        start_idx = 0
        left = 1
        right = 2
        while(len(self.VOCAB) - start_idx != 1):
            rep_vector = torch.randn(self.REP_VECTOR_SIZE, requires_grad=True, device = self.DEVICE)
            self.PARAMS.append(nn.Parameter(rep_vector))
            root = self.VOCAB[start_idx]
            minimum = None
            min_idx = None
            #nodes with low frequency get combined into larger frequency nodes and reheapify
            if(right < len(self.VOCAB) and self.VOCAB[left].FREQ > self.VOCAB[right].FREQ):
                minimum = self.VOCAB[right]
                min_idx = right
            else:
                right_node = None
                minimum = self.VOCAB[left]
                min_idx = left
            #freq, word, rep_vector, left, right
            new_freq = root.FREQ+minimum.FREQ
            self.VOCAB[min_idx] = Node(new_freq,"Internal Node",rep_vector,root, minimum)
            start_idx += 1
            left += 1
            right += 1
            self.heapify(start_idx)
        self.ROOT = self.VOCAB[-1]
        
    def assign_codes(self,root,path):
        #postorder path code assignment
        if(root and root.left):
            self.assign_codes(root.left,path+[0])
        if(root and root.right):
            self.assign_codes(root.right,path+[1])
        if(root.WORD != None):
            self.PATHS.update({root.WORD:path})
    def get_prob(self,word_vec):
        #traverse grab the word
        root = self.ROOT
        prob = 1
        while(root.REP_VECTOR != None):
            prob *= torch.sigmoid(torch.matmul(root.REP_VECTOR,word_vec))
            if(prob >= 0.5):
                root = root.right
            else:
                root = root.left
        return root.WORD
    def train(self,word_vec,word):
        #traverse grab the probability
        prob = 1
        root = self.ROOT
        if word not in self.VOCAB_DICT:
            raise MissingValueError("Word not in vocab")
        path = self.PATHS[word]
        for digit in path:
            right_prob = torch.sigmoid(torch.matmul(root.REP_VECTOR,word_vec))
            if digit == 1:
                prob *= right_prob
                root = root.right
            else:
                prob *= (1-right_prob)
                root = root.left
                
        return prob
    def to(self, device):
        for param, i in enumerate(self.PARAMS):
            self.PARAMS[i] = param.to(device)
    #print tree method
    #source: geeks4geeks copied for debugging   
    def print_util(self, root, space):
        if (root == None) :
            return
    
        space += 10
    
        self.print_util(root.right, space)
    

        print()
        for i in range(10, space):
            print(end = " ")
        print(root.WORD)
    
        self.print_util(root.left, space)
    def print_tree(self):
        self.print_util(self.ROOT,0)
    