# Hierarchical Probabilistic Ultrasound Image Inpainting via Variational Inference

Official code for paper [Hierarchical Probabilistic Ultrasound Image Inpainting via Variational Inference](google.com)

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
```