# Comparison of FID and Inception Score

This experiments should highlight a crucial difference between the FID and the Inception Score (IS).
The purpose of a generative model is to learn a real world distribution. Thus a good performance measure
should, roughly speaking, somehow capture how far off the model distribution is. The experiments show,
that in this sense the FID is a more useful measure.

## Methodology
While the idea of the IS is to capture 1) how real the structures in the generated images are and
2) how much variability the generated samples have, there is no connection of the score to the
real world distribution. Clearly the assumptions of the IS are met best on the dataset it is trained
on, namely the ImageNet data set. It is however questionable, if the assumptions carry over to another image
datasets. As an example consider the celebA dataset. It consists of about 200k face images of celebrities.
While assumption 1) still holds it is not so clear why there should be a high variability across samples.

But the main point is: an evaluation method should indicate how well the real world
distribution has been learned. This implies: disturbed images should lead to a
lower score or a higher distance respectively. Thus for the experiments we produce
disturbed images of the celebA dataset with increasing disturbance levels
to evaluate the FID and IS on them.
The IS is transformed to an distance as described in the TTUR paper. This is done to
make comparison between the two methods easier. We refer to the transformed
IS as the IND - the inception distance.

## Experiments
1. Gaussian noise: We constructed a matrix N with Gaussian noise scaled to [0, 255]. The
noisy image is computed as (1 − α)X + αN for α ∈ {0, 0.25, 0.5, 0.75}. The larger α is,
the larger is the noise added to the image, the larger is the disturbance of the image.  

|FID|IND|
|-|-|
|<img src=figures/png/gnoise_FID.png width=350 height=350 /> |<img src=figures/png/gnoise_IND.png width=350 height=350 />|

2. Gaussian blur: The image is convolved with a Gaussian kernel with standard deviation
α ∈ {0, 1, 2, 4}. The larger α is, the larger is the disturbance of the image, that is,
the more the image is smoothed.

|FID | IND|
|-|-|
|<img src=figures/png/blur_FID.png width=350 height=350 /> |<img src=figures/png/blur_IND.png width=350 height=350 />|


3. Black rectangles: To an image five black rectangles are are added at randomly chosen
locations. The rectangles cover parts of the image.The size of the rectangles is
α imagesize with α ∈ {0, 0.25, 0.5, 0.75}. The larger α is, the larger is the disturbance
of the image, that is, the more of the image is covered by black rectangles.   

|FID|IND|
|-|-|
|<img src=figures/png/rect_FID.png width=350 height=355 />| <img src=figures/png/rect_IND.png width=350 height=355 />|


4. Swirl: Parts of the image are transformed as a spiral, that is, as a swirl (whirlpool
effect). Consider the coordinate (x, y) in the noisy (swirled) image for which we want to
find the color. Toward this end we need the reverse mapping for the swirl transformation
which gives the location which is mapped to (x, y). The disturbance level is given by the
amount of swirl α ∈ {0, 1, 2, 4}. The larger α is, the larger is the disturbance of the
image via the amount of swirl.                                                              

|FID|IND|
|-|-|
|<img src=figures/png/swirl_FID.png width=350 height=350 /> | <img src=figures/png/swirl_IND.png width=350 height=350 />|


5. Salt and pepper noise: Some pixels of the image are set to black or white, where black is
chosen with 50% probability (same for white). Pixels are randomly chosen for being flipped
to white or black, where the ratio of pixel flipped to white or black is given by the noise
level α ∈ {0, 0.1, 0.2, 0.3}. The larger α is, the larger is the noise added to the image via
flipping pixels to white or black, the larger is the disturbance level.  

|FID|IND|
|-|-|
|<img src=figures/png/sp_FID.png width=350 height=350 /> | <img src=figures/png/sp_IND.png width=350 height=350 />|


6. ImageNet contamination: From each of the 1,000 ImageNet classes, 5 images are randomly
chosen, which gives 5,000 ImageNet images. The images are ensured to be RGB and to
have a minimal size of 256x256. A percentage of α ∈ {0, 0.25, 0.5, 0.75} of the CelebA
images has been replaced by ImageNet images. α = 0 means all images are from CelebA,
α = 0.25 means that 75% of the images are from CelebA and 25% from ImageNet etc.
The larger α is, the larger is the disturbance of the CelebA dataset by contaminating it by
ImageNet images. The larger the disturbance level is, the more the dataset deviates from the
reference real world dataset.

|FID|IND|
|-|-|
|<img src=figures/png/mixed_FID.png width=350 height=350 /> | <img src=figures/png/mixed_IND.png width=350 height=350 />|
