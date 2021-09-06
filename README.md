# **Pytorch Hierarchical Softmax Extension**<br/>
Hierarchical softmax is an efficient alternative to the popular softmax activation function that can be computed in O(log(N)) time instead of the usual O(N) time by breaking the probability computation into a traversable binary tree as illustrated below. For this reason, it is quite popular in NLP tasks to save time.<br/>

![image](https://user-images.githubusercontent.com/63683831/132118973-2b10e5f1-7543-4439-b0f0-407d6ac67118.png)<br/><br/>

###### **Source: BuildingBabylon** <br/><br/>


Pytorch does not have an in-built extension for hierarchical softmax, so I built a python package for that. It can be treated as a Pytorch module in that it is easy to integrate into Pytorch's models.<br/>

This implementation in particular uses a Huffman tree as the tree type to determine the probabilities of words.

To install, do ```pip install <REPO LINK>```.<br/><br/>
Once installed, import the package with ```import hier_softmax```.<br/><br/>


Initialize the tree with ```hier_softmax(vocab, vector_size, device)```. <br/>
The vocab is a dictionary of all the words in the vocabulary and their frequencies. <br/>
The vector size is the size of the representation vectors that will be on the tree(explained in paper describing implementation linked below). <br/>
The device is the cuda device on which the parameters will be allocated.<br/><br/>


To get the probabilities of words pass the sequences to the ```get_prob(seq_list)``` method, which will find the word probabilities. To train the tree, ```train(seq_list, word_list)``` can be used. <br/>
The ```seq_list``` and ```word_list``` parameter is described in the ```hier_softmax.py``` file. <br/>

Everything else functions the same as a regular Pytorch module.<br/><br/><br/><br/>




Paper: https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
