# Hierarchical Probabilistic Ultrasound Image Inpainting via Variational Inference

Official code for paper [Hierarchical Probabilistic Ultrasound Image Inpainting via Variational Inference](https://link.springer.com/content/pdf/10.1007%2F978-3-030-88210-5_7.pdf), which was accepted at DGM4MICCAI 2021.

[Slides](https://drive.google.com/file/d/1QiznJ6UjWqhXJ35T8szzukaGrJgkLQDz/view?usp=sharing) and [Presentation](https://www.youtube.com/watch?v=7jEW8chI4QA)

# Model Initialization
Our implementation follows the following parameters of the model
```python
	model = HierarchicalProbUNet(
        num_layers=6,
        num_filters=[64,128,256,512,1024,1024],
        num_prior_layers=4,
        num_filters_prior=[5,5,5,5],
        dilation=[1,1,2,2,2,4],
        p=[0,0,0,0,0.01],
        s=[0.1,0.002,0.001,0.01,10],
        rec=100,
        tv=0,
        additional_block='CBAM',
        name='ProbUNet',
    	)
	inputs=tf.keras.Input(shape=(256,256,4,))
	model(inputs)
```


# Credit 
If you use the code or the paper in any of your work, please remember to cite us
```bash
@incollection{hung2021hierarchical,
  title={Hierarchical Probabilistic Ultrasound Image Inpainting via Variational Inference},
  author={Hung, Alex Ling Yu and Sun, Zhiqing and Chen, Wanwen and Galeotti, John},
  booktitle={Deep Generative Models, and Data Augmentation, Labelling, and Imperfections},
  pages={83--92},
  year={2021},
  publisher={Springer}
}
```