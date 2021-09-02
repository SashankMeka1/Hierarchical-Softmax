import torch
import torch.nn as nn
import node
class HuffmanTree(nn.Module):
    def __init__(self,vocab_dict,rep_vector_size, device):
        super().__init__()
        self.root = None
        self.paths = {}
        self.params = []
        self.device = device
        self.rep_vector_size = rep_vector_size
        self.vocab_dict = vocab_dict
        self.vocab = []
        for word,freq in list(vocab_dict.items()):#build heap of nodes
            self.vocab.append(node(freq,None,None,None).set_word(word))
        self.build_heap()
        self.assign_codes(self.root,[])#assign codes
        self.params = nn.ParameterList(self.params)
    def heapify(self,idx):
        minimum = idx
        left = (2*idx)+1
        right = (2*idx)+2

        #recursive heapify smaller frequency nodes first with higher freq as children
        if(left < len(self.vocab) and self.vocab[left].freq < self.vocab[minimum].freq):
            minimum = left
        if(right < len(self.vocab) and self.vocab[right].freq < self.vocab[minimum].freq):
            minimum = right
        if minimum != idx:
            swap(self.vocab[minimum],self.vocab[idx])
            heapify(minimum)
    def build_heap(self):
        for idx in range(len(self.vocab)//2):
            heapify(idx)
    def build_huffman_tree(self):
        start_idx = 0
        left = 1
        right = 2
        while(len(self.vocab) - start_idx != 1):
            rep_vector = torch.randn(self.rep_vector_size, requires_grad=True, device = self.device)
            self.params.append(nn.Parameter(rep_vector))
            root = self.vocab[start_idx]
            left_node = self.vocab[left]
            if(right < len(self.vocab)):
                right_node = self.vocab[right]
            else:
                right_node = None
            #nodes with low frequency get combined into larger frequency nodes and reheapify
            if(right_node == None and left_node.freq < right_node.freq):
                self.vocab[left] = node(root.freq+left_node.freq,rep_vector,root,left_node)
            elif(left_node.freq > right_node.freq):
                self.vocab[right] = node(root.freq+right_node.freq,rep_vector,root,right_node)
            start_idx += 1
            left += start_idx
            right += start_idx
            heapify(start_idx)
        
    def assign_codes(self,root,path):
        #postorder path code assignment
        if(root.left):
            assign_codes(root.left,path+[0])
        if(root.right):
            assign_codes(root.right,path+[1])
        if(root.word != None):
            self.paths.update({word:path})
    def get_prob(self,word_vec):
        #traverse grab the word
        root = self.root
        prob = 1
        while(root.rep_vector != None):
            prob *= torch.sigmoid(torch.matmul(root.rep_vector,word_vec))
            if(prob >= 0.5):
                root = root.right
            else:
                root = root.left
        return root.word
    def train(self,word_vec,word):
        #traverse grab the probability
        prob = 1
        root = self.root
        if word not in self.vocab_dict:
            raise MissingValueError("Word not in vocab")
        path = self.paths[word]
        for digit in path:
            right_prob = torch.sigmoid(torch.matmul(root.rep_vector,word_vec))
            if digit == 1:
                prob *= right_prob
            else:
                prob *= (1-right_prob)
        return prob
    def to(self, device):
        for param, i in enumerate(self.params):
            self.params[i] = param.to(device)

        

        
