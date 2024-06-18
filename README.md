# Image_denoising_zero_dce_net
PSNR: 28.07

Spatial Consistency Loss: By ensuring consistency of the spatial feature maps produced over consecutive training epochs, maintaining per-class running-average heatmaps for each training image. We show that this spatial consistency loss further improves the enhanced image through preserving the difference of neighboring regions between the input image and its enhanced version.
![image](https://github.com/krishagrwl/Image_Denoising_Zero_DCE_NET/assets/172372978/d67ff76b-9d35-4c2d-845b-31a23ea14320)
Color Constancy Loss: This function aims to ensure that the enhanced image maintains a consistent color balance. This function calculates the loss based on the mean color values of the red, green, and blue channels of the image, penalizing deviations from an ideal color balance. Then the function calculates the squared differences between the mean value of each pair of channels. Function returns square root of sum of differences of each pair of means.
The use of square root ensures that the loss function is differentiable and provides a gradient that can be used for optimization.
![image](https://github.com/krishagrwl/Image_Denoising_Zero_DCE_NET/assets/172372978/4c42ddaf-fac4-41b0-a103-77173fcdf8fb)
Exposure Loss: This function is designed to ensure that the enhanced image has an appropriate overall brightness level. This function calculates the loss based on how the mean brightness of overlapping patches of the image deviate from specific target value. First we reduce number of channels to 1 by taking average of R, G, B channels which represents average brightness value of that pixel. Then function apply Average pooling over non overlapping patches which computes 
![image](https://github.com/krishagrwl/Image_Denoising_Zero_DCE_NET/assets/172372978/2d2139ab-d038-46d7-832b-a01671479e30)

Illumination Smoothness Loss: This function is essential in improving images using deep learning. It focuses on making sure that changes in brightness across the image are smooth and gradual, rather than sudden and jarring. It looks at batches of images and calculates how much each pixel's brightness differs from its neighbors vertically and horizontally. By averaging these differences across all pixels and images in the batch, the function helps create enhanced images that look more natural and pleasing to the eye. This is particularly useful in tasks where maintaining a consistent look throughout the image is important for overall image quality.
![image](https://github.com/krishagrwl/Image_Denoising_Zero_DCE_NET/assets/172372978/9269e716-a011-475f-920f-e9a7011fcce9)

•	zero_dce_model.py: This python file contains architecture of model 
•	losses.py: This python file contains all the loss function which is to be used in model 
•	train.ipynb: This file runs the training script and saves a model.h5 file in current directory 
•	main.ipynb: This file runs the inference on image which are present in test/low/ of current directory and save results on /test/predicted. 
RESOURCES:
•	Zero DCE Paper: https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf
