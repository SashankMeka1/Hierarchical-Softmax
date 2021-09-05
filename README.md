# **Pytorch Hierarchical Softmax Extension**<br/>
Hierarchical softmax is a softmax alternative that can be computed in O(log(n)) time instead of the usual O(n) time. For this reason, it is quite popular in NLP tasks to save time.
Pytorch does not have an in-built extension for hierarchical softmax, so I built a python package for that. It can be treated as a pytorch module in that it is easy to integrate into Pytorch's models.

To install, do ```pip install <REPO LINK>```.
