---
author: Luis Nicolás Luarte Rodríguez
title: Tracking Software
output: revealjs::revealjs_presentation
---

# Main idea

- At each time step:
	- Separate background from foreground
	- Get the head point
	- Get the "body" point
	- Get the tail point
	- Do further analysis with those points

# Ideal input

- First step is to have a good idea of what our background is
	- We can create a model (more, complex but better suited for environments with dynamic lighting)
	- Or we can assume that out background is never going to change (how most labs do it)

# We can get an image of our background

![Example background](/home/nicoluarte/uni/PHD/tracking_device/backpic.png)

# However a simple image is rather complex, because each pixel has 3 channels (red, green, blue)

![RGB channels](/home/nicoluarte/uni/PHD/tracking_device/pixels.png){ width=15% }

# To simplify we turn it to gray scale (only 1 channel)

![Grayscale = (r + g + b / 3)](/home/nicoluarte/uni/PHD/tracking_device/gray.png){ width=15% }

# However, we need to further process our background to make it 'smoother'

- We need to remove noise
	- Noise are 'details' that make an image harder to identify
	- In other words, denoising is equivalent to make an image more homogeneous, while preserving edges
	- A bilateral filter does such thing

# Noise image

![Notice the details](/home/nicoluarte/uni/PHD/tracking_device/noise-png.png){ width=40% }

# Noise image after bilateral filter

![Details are gone, edges preserved](/home/nicoluarte/uni/PHD/tracking_device/denoise.png){ width=40% }

# Similar steps are applied to the image with the animal

![Smoothed image](/home/nicoluarte/uni/PHD/tracking_device/preproc.png){ width=50% }

# The next step is to substract the background from the image with the animal

- This is done by substracting pixel intensity
- The result is not good, some areas of the animal are considered background (black color)

![Difference](/home/nicoluarte/uni/PHD/tracking_device/diff.png){ width=50% }

# We use morphological transformations to fix this

- Morphological transformations are operations applied to binary images, which are based on the image shape
	- They use a 'kernel', which a windows where a certain operation is performed
- Our main problem is that there's background objects INSIDE the animal
	- The closing operation is applied to the kernel
		- A pixel element is defined as '1' if: inside the kernel there's atleast a '1' pixel (dilation)
		- Then we 'erode' the boundaries: a pixel is considered '1' is all pixels under the kernel are 1's, otherwise is 0

# Dilation

![](/home/nicoluarte/uni/PHD/tracking_device/image.png){ width=25% }
![](/home/nicoluarte/uni/PHD/tracking_device/dilation.png){ width=25% }

# Erosion

![](/home/nicoluarte/uni/PHD/tracking_device/image.png){ width=25% }
![](/home/nicoluarte/uni/PHD/tracking_device/erosion.png){ width=25% }

# Closing = dilation followed by erosion

![Closing](/home/nicoluarte/uni/PHD/tracking_device/closing.png){ width=50% }

# Result

![Not perfect, but the animal is clearly isolated](/home/nicoluarte/uni/PHD/tracking_device/morphological.png){ width=50% }

# The original image is not ideal, but we can improve this and make it a more robust algorithm

- We can calculate the contours of the image
	- This makes it more robust against 'lighting artifacts'

![Image with artifact](/home/nicoluarte/uni/PHD/tracking_device/foreground2.png){ width=50% }

# Artifact removal

- The algorithm simply calculates the contours of each objects, and selects the one with the bigger area
	- The image looks worse because the artifact is super big
	- In normal conditions an artifact like that can be removed manually

![Image without artifact](/home/nicoluarte/uni/PHD/tracking_device/noartifact.png){ width=50% }

# After the segmentation problem, we need to find the head, body and tail

- The intuition is:
	- Rats/mice are 'chubby'
	- Long tails
	- 'small' head relative to the body
	- The tail is the furthest away point from the body
	- The head is the furthest away point from the tail

# The geodesic distance is the proper implementation to solve this

- The geodesic distance is a shortest path between 2 points in a certain space
	- We define our space as the animal surrounded by boundaries (background)
	- The distance, considering the boundaries, is the geodesic distance
	- Is approximated by the fast marching algorithm

# Geodesic vs euclidean distance

![Notice how the boundaries inform the distance at the feet](/home/nicoluarte/uni/PHD/tracking_device/geodesic.png){ width=50% }

# Considering this, the furthest away point is the 'body' because the animal is 'chubby'

![Warmer color are more distant](/home/nicoluarte/uni/PHD/tracking_device/points.png){ width=50% }

# Detecting the head and the tail is relatively simple

![In read the calculated points](/home/nicoluarte/uni/PHD/tracking_device/final.png){ width=50% }

# The final algorithm should be (not tested in real animals yet) resistant to rotations, minimal dynamics in lighting conditions and non-uniform backgrounds

show video

# How to implement the system

![The big picture](/home/nicoluarte/uni/PHD/tracking_device/bigpicture.png){ width=65% }

# TODO list

- Test with real animals
- Implement the GUI (only few functions right now)
- Test with multiple pi
- Stabilize frame-rate
- Pack and deploy into a docker (possibly)

