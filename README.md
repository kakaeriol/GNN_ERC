# Emotion Recognition In Conversation with <br> Transformer and Relational Graph Transformer

In this project, based on the DialogueGCN  [[1]](#1), we proposed new flows the have better performance than the original paper. 

In the DialogGCN, their flow is as show in the below image

![GCN drawio](https://user-images.githubusercontent.com/16068098/232380690-e7ecb8a5-4419-4b92-a159-90ddc51d1bf1.png)

In our methods, we changed the sequence context to Transformer, and RGCN to Relational Graph Transform, which better generalize and have global information. 

![RGT](https://user-images.githubusercontent.com/16068098/232379831-352d43c3-2e8e-4c27-af18-50948ddf8421.png)

In our methods, Relational Graph Tranformer having the same properties as Graph Transformer Architecture [[2]](#2) but include the information of the relation after multiheader layer 

![image](https://user-images.githubusercontent.com/16068098/232381202-79548802-01bb-4649-a3f0-c39bbfa02ad7.png)


For the coding, we use DGL library [[3]](#3)
- The full raw dataset here: https://drive.google.com/drive/u/2/folders/1LFd3KbxhwxSrth6MqKlIa-uB-iskvSwZ </br>
- The data after preprocesssing here: https://drive.google.com/drive/u/2/folders/1js4MoIQDYPSa62DpwwGsyJyt3oV4cN7n </br>


If you want to check the process of preprocessing data, extract the raw data in the folder and check change the path in sysconf and run the script in preprocessing folder

If you want to recheck the training without preprocessing steps, please download the full preprocessing data as the link above, extract and change the path in sysconf folder. After that enter our_method folder or base_line folder and try to run the code.


## Reference

<a id = "1">[1]</a>
Deepanway Ghosal, Navonil Majumder, Soujanya Poria, Niyati Chhaya,
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
}</br>
<a id = "3">[3]</a>
https://www.dgl.ai/
