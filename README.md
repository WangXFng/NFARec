## NFARec (SIGIR'24) [arxiv](https://arxiv.org/pdf/2404.06900).

## Paper
- NFARec: A Negative Feedback-Aware Recommender Model, **SIGIR 2024 Oral**.
- [**Xinfeng Wang**](https://wangxfng.github.io/), Fumiyo Fukumoto, Jin Cui, Yoshimi Suzuki, Dongjin Yu.


### Run
	python Main.py

### Note
* Configures are given by Constants.py and Main.py
* If you have any problem, please feel free to contact me at kaysenn@163.com.

### Dependencies
	pip install -r requirement.txt
___

### Datasets
	Three files are required: train.txt (for training), tune.txt (for tuning), and test.txt (for testing).
	Each line denotes an interaction including a user rated an item with a score at times (or timestamp).
	The format is [#USER_ID]\t[#ITEM_ID]\t[#SCORES]\t[#TIMES]\n, which is the same for all files.
	For example,
	0	0	5	1
	0	1	4	3
	0	3	1	2
	1	2	4	1
	the user (ID=0) rated the item (ID=0) with the score of 5 at 1 time (or timestamp), 
				  the item (ID=1) with the score of 4 at 3 times (or timestamp), 
				  and the item (ID=3) with the score of 1 at 2 times  (or timestamp).
	the user (ID=1) rated the item (ID=2) with the score of 4 at 1 time (or timestamp).


### Citation
If this repository helps you, please cite:

	@inproceedings{wang2024nfarec,
	  title={NFARec: A Negative Feedback-Aware Recommender Model},
	  author={Wang, Xinfeng and Fukumoto, Fumiyo and Cui, Jin and Suzuki, Yoshimi and Yu, Dongjin},
	  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
   	  pages={935–-945},
	  year={2024}
	}
