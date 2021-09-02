from huffman.huffman import HuffmanTree
import torch
import torch.nn as nn
class HierSoftmax(nn.Module):
    #pass dictionary of word frequency/size of rep vectors/cuda device
    def __init__(self,vocab,vector_size, device):
        self.__tree = HuffmanTree(vocab,vector_size, device)
    def to(self, device):
        self.__tree.to(device)
    def check_error(**kwargs):
        if("seq_list" in kwargs):
            if(len(kwargs["word_list"]) == 0):
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
        check_error(seq_list = seq_list)
        for batch,batch_idx in enumerate(seq_list):
            for word_vec,word_idx in enumerate(batch):
                word_matrix[batch_idx][word_idx] = self.__tree.get_prob(word_vec)
        return seq_list
    
    #pass list of sequences with same requirements as get_prob
    #receive loss for entire batch
    def train(self,seq_list,word_list):
        check_error(seq_list = seq_list, word_list = word_list)
        batch_error = []
        for batch,batch_idx in enumerate(seq_list):
            seq_error = []
            for word_vec,word in zip(batch, word_list[batch]):
                sample_ce_loss = -torch.log(self.__tree.train(word_vec,word_list[word_idx]))
                seq_error.append(sample_ce_loss)
            batch_error.append(torch.mean(torch.FloatTensor(seq_error)))
        return torch.mean(torch.FloatTensor(batch_error))
