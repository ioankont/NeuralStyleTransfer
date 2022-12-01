# Neural-algorithm-of-Style-Transfer in Tensorflow 2

Implementation of Image Style Transfer using Convolutional Neural Networks from the paper (L.A.Gatys, A.S.Ecker and M.Bethge, [2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html))

## How about mixing an image with the style of your favorite painter?

Choosing an image as a content and a style image, you can combine them and here are the results.

  
<p align="center">
  Content Image
</p>
<p align="center">
  <img src="https://github.com/ioankont/NeuralStyleTransfer/blob/main/pictures/ContentImages/Lion.jpg" width="256" height="256" title="Content Image">  
</p>



<p float="left">
  <p align="center">
    Style Images
    </p>
    <p align="center">
    <img src="https://github.com/ioankont/NeuralStyleTransfer/blob/main/pictures/StyleImages/Portrait.jpg" width="256" height="256" title="Style Image 1"> 
    <img src="https://github.com/ioankont/NeuralStyleTransfer/blob/main/pictures/StyleImages/Scream.jpg" width="256" height="256" title="Style Image 2"> 
    <img src="https://github.com/ioankont/NeuralStyleTransfer/blob/main/pictures/StyleImages/TheMuse.jpg" width="256" height="256" title="Style Image 3"/>
    </p>
</p>

<p float="left">
   <p align="center">
    Results
   </p>
   <p align="center">
    <img src="https://github.com/ioankont/NeuralStyleTransfer/blob/main/pictures/Results/Lion_Portrait.jpg" width="256" height="256" title="Style Image 1"> 
    <img src="https://github.com/ioankont/NeuralStyleTransfer/blob/main/pictures/Results/Lion_Scream.jpg" width="256" height="256" title="Style Image 2"> 
    <img src="https://github.com/ioankont/NeuralStyleTransfer/blob/main/pictures/Results/Lion_TheMuse.jpg" width="256" height="256" title="Style Image 3"/>
  </p>
</p>

## Description
It is a Style Transfer algorithm, and since the texture and style are also based on the deep representations of the images, it turns into an optimization problem, where the feature representations of the new image will be derived to match the features of the content image (content image) and style image (style image). <br />
<br />
Used a pretrained model VGG19 to extract the feauture representations of the content and the style images. We build our result in a white-noise image. <br />
For content representations, it was showned that in a trained CNN in object detection, we can represent the semantic information of the input image in a layer, reconstructing the image with the feauture maps of this layer. Reconstructing in low-lever layers, would reuslt in the exact pixel values, while in high-lever layers the objects are represented without limiting the pixel values. <br />
For style representations, used Gram Matrix which calculate the correlations between different responses. As a result we extract the style feautures of an image. Included the feauture correlations of different layers of the network, so that the result consists of a multi-scale representation of texture informations.

### Algorithm <br />
#### *Lcontent*
A white-noise image _x_ passes through the VGG19 and we extract his content representation from block4_2. Similarly, from the same layer for content image *p*. We calculate *Lcontent* so that the feature represantion of the *x* matching to *p* with the use of gradient descent. <br />
#### *Lstyle*
For *x* and the style image *a* we extract the style representation from block1_1, block2_1, block3_1, block4_1, block5_1. We calculate the Gram Matrices for all layers and as a result it occurs the function *Lstyle*. Reducing this function, constructing an image *x* that matches the style representation of the style image *a*.
#### *Lloss* = α*Lcontent* + β*Lstyle* <br />
<p align="center">
  <img src="https://user-images.githubusercontent.com/118340733/205013101-d8ead630-6664-4024-a419-3ef075c3f5cf.JPG" width="800" height="400" title="Content Image">
</p> <br />
The unusual in this algorithm is that we do not update the weights of a model, but the values of the white-noise image *x*


 
