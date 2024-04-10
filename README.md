Segment-Everything is a CNN-based segmentation workflow for multi-phase semantic segmentation using a few 2D slices. The segmentation network includes U-ResNet and Efficient-Unet. Please ensure all the required libraries are installed, and GPU is available.

User manual for Segment-Everything:
1. The CNN-based segmentation is trained based on multiple 2D paired grayscale images and labeled images. Therefore, the user has to prepare these training pairs. We suggest using trainable Weka segmentation to
   generate the 2D pairs.
2. After generating these 2D pairs, save them into the file "training-pairs", grayscale images are saved to "training-pairs/raw", and labeled images are saved to "training-pairs/seg".
3. Run the trainingdataPrepare.py, which will create training patches. And create the required file directory. Ensure you change the directory in the .py file.
   Some user-defined parameters can be modified, including patch size (default is 96x96), and number of patches (default is 800 from one 2D slice).
5. Run the train.py, which will train the CNN model. A few of the hyperparameters can be changed by users, such as epoch, learning rate, batch size, and network types. Ensure you change the directory in the .py file.
6. Once the train.py is finished, the trained models are saved to the "ck" file. Run the test.py to load the models to segment either a 2D slice or a 3D domain.


An example of a 3-phase sandstone image segmentation is provided. One training pair is stored in "training-pairs" as well as a testing 2D slice is stored in "test/2D". 


Please cite the following papers if use the code:
https://doi.org/10.1103/PhysRevApplied.17.034048;

https://doi.org/10.1016/j.compchemeng.2022.107768;

https://doi.org/10.1016/j.mineng.2022.107592;
