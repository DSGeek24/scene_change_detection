Detecting the difference in scenes using Least Squares, Gaussian and Robust Estimators fitting models. Assume two images I1 and I2 where I1(x,y) represents the pixel intensity of the image at location (x,y) and similarily for I2(x,y). When two images are equal the plot would be symmetrical in a Euclidean space. When there is a deviation, data points which have undergone major changes represent the pixels which are different and hence we can find a model using fitting approach to discover such pixels. This model can then be used to find pixels that are significantly different from a previous scene.  

Exaple usage:

python scene_change_detection.py -i1 image1 -i2 image2 -m TL -t 20(Least square fitting model)

python scene_change_detection.py -i1 image1 -i2 image2 -m RO -t 20(Robust estimation model)

python scene_change_detection.py -i1 image1 -i2 image2 -m GA -t 20(Gaussian Fitting model)

Output images or files will be saved to "output/" folder (scene_change_detection.py automatically does this)
  
-------------

