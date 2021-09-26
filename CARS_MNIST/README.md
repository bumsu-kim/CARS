MNIST Attack using CARS and Square-Attack

Usage: Run MNIST_Attack.py.
This script loads the pre-trained model and attacks the images.
Sample Usage: python MNIST_Attack.py -o CARS -name "CARS_dir_name" -v 1 -eps 0.3 -si 0.3 -t 0 10


- A pretrained model is included. When you run MNIST_Attack.py, it will automatically load it.
- MNIST Dataset is also included. Adversarial attacks will be done to testset images.
- DEFINITION of TID (testset id):
   Label "n" in TID "t" means it is the "t"-th "n" appearing in the test set.
   And this image(as a 1-D np array) is stored in "atk_test_set[t][n]"
   So, for instance, "atk_test_set[0]" contains 10 images,
     which are the first 0, first 1, ..., first 9 in the test set.
     
 
