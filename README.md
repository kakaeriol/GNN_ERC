# Emotion Recognition In Conversation with Transformer and Relational Graph Transformer

In this project, based on the DialogueGCN  [[1]](#1), we proposed new flows the have better performance than the original paper. 

In the DialogGCN, their flow is as show in the below image
![GCN drawio](https://user-images.githubusercontent.com/16068098/232380690-e7ecb8a5-4419-4b92-a159-90ddc51d1bf1.png)

In our methods, we changed the sequence context to Transformer, and RGCN to Relational Graph Transform, which better generalize and have global information. 

![RGT](https://user-images.githubusercontent.com/16068098/232379831-352d43c3-2e8e-4c27-af18-50948ddf8421.png)

In our methods, Relational Graph Tranformer having the same properties as Graph Transformer Architecture [[2]](#2)

For the coding, we use DGL library [[3]](#3)

## Reference

<a id = "1">[1]</a>
eepanway Ghosal, Navonil Majumder, Soujanya Poria, Niyati Chhaya,
and Alexander Gelbukh. Dialoguegcn: A graph convolutional neural
network for emotion recognition in conversation. In Proceedings of
the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural
Language Processing (EMNLP-IJCNLP), pages 154–164, 2019. <br>

<a id = "2">[2]</a>
@article{dwivedi2021generalization,
  title={A Generalization of Transformer Networks to Graphs},
  author={Dwivedi, Vijay Prakash and Bresson, Xavier},
  journal={AAAI Workshop on Deep Learning on Graphs: Methods and Applications},
  year={2021}
}
<a id = "3">[3]</a>
https://www.dgl.ai/
