# MNIST-NeuralNet
This is a Neural Network i made with keras  
I included a pretrained model and some handwritten ditits  
If you don't want to use the GUI just replace it with custom(),  
which will load your custom images (which have to be named in the format the included ones are)  
If you want to train the model just call train_model()  
In the GUI version I made it so that every digit is shown as the model sees it after you press "Recognize"    
If you don't want that then you can just remove the call to show_img() in classify_handwriting()  