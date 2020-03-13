##important files are not uploaded...waiting for permission from Tencent
# grec

@article{yuan2019future,
  title={Future Data Helps Training: Modeling Future Contexts for Session-based Recommendation},
  author={Yuan, Fajie and He, Xiangnan and Jiang, Haochuan and Guo, Guibing and Xiong, Jian and Xu, Zhezhao and Xiong, Yilin},
  journal={arXiv},
  pages={arXiv--1906},
  year={2019}
}

python GRec.py

or
python GRec_NCE.py  (NCE sampling. Be careful with the number of negative examples.)

You can also replace our sampling method and CE loss with other well-known negative sampler or loss functions, such as LambdaFM.



