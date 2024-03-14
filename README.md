Segment-Everything is a CNN-based segmentation workflow for multi-phase semantic segmentation using a few 2D slices. The segmentation network includes U-ResNet and Efficient-Unet.

User manual for Segment-Everything:
1. The CNN-based segmentation is trained based on multiple 2D paired grayscale images and labeled images. Therefore, the user has to prepare these training pairs. We suggest using trainable Weka segmentation to
   generate the 2D paires.
2. After generating the 2D pairs, save them into the file "training-paires", grayscale images are saved to "training-paires\raw", and labeled images are saved to "training-paires\seg".
3. Run the trainingdataPrepare.py, which will create training patches. And create the required file directory. Ensure you change the directory in the .py file.
4. Run the train.py, which will train the CNN model. A few of the parameters can be changed by users, such as epoch, learning rate, batch size, and networks. Ensure you change the directory in the .py file.
5. Once the train.py is finished, the trained models are saved to the "ck" file. Run the test.py to load the models to segment either a 2D slice or a 3D domain.
