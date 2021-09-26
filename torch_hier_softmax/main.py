from torch_hier_softmax.huffman.huffman import HuffmanTree
import torch
import torch.nn as nn
class HierSoftmax(nn.Module):
    #pass dictionary of word frequency/size of rep vectors/cuda device
    def __init__(self,vocab,vector_size, device):
        super(HierSoftmax, self).__init__()
        self.tree = HuffmanTree(vocab,vector_size, device)
    def to(self, device):
        self.tree.to(device)
    def check_error(*args, **kwargs):
        if("seq_list" in kwargs):
            if(len(kwargs["seq_list"]) == 0):
                raise ValueError("Empty list of sequences")

            if("word_list" in kwargs):
                if(len(kwargs["word_list"]) != len(kwargs["seq_list"])):
                    raise ValueError("Size mismatch: sequences of x and y are different sizes")
        else:
            raise ValueError("Missing Sequence list")
            

    #get the probability of a word pass as list of sequences
    #each sequence should contain list of vectors representing words
    #will return list of sequences with word for each vector
    def get_prob(self,seq_list):
        self.check_error(seq_list = seq_list)
        out_list = []
        for batch in seq_list:
            out_list.append([])
            for word_vec in batch:
                out_list[-1].append(self.tree.get_prob(word_vec))
        return out_list
    
    #pass list of sequences with same requirements as get_prob
    #receive loss for entire batch
    def train(self,seq_list,word_list):
        self.check_error(seq_list = seq_list, word_list = word_list)
        batch_error = 0
        batch_num = 0
        for seq_idx,seq in enumerate(seq_list):
            words = word_list[seq_idx]
            seq_error = 0
            seq_num = 0
            for word_idx,word_vec in enumerate(seq):
                sample_ce_loss = -torch.log(self.tree.train(word_vec,words[word_idx]))
                seq_error += sample_ce_loss
                seq_num += 1
            batch_error += seq_error
            batch_num += 1
        return batch_error/batch_num
