# **Pytorch Hierarchical Softmax Extension**<br/>
Hierarchical softmax is an efficient alternative to the popular softmax activation function that can be computed in O(log(N)) time instead of the usual O(N) time. For this reason, it is quite popular in NLP tasks to save time.<br/>
![image](https://user-images.githubusercontent.com/63683831/132118563-31ea07ac-34ac-4e74-b0ad-955dde4e118f.png)<br/>
Pytorch does not have an in-built extension for hierarchical softmax, so I built a python package for that. It can be treated as a Pytorch module in that it is easy to integrate into Pytorch's models.<br/>
This implementation in particular uses a Huffman tree as the tree type to determine the probabilities of words.

To install, do ```pip install <REPO LINK>```.
Once installed, import the package with ```import hiersmax as HierSoftmax```
