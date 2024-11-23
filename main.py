import os
from xmlrpc.client import Boolean

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
from skimage.morphology import remove_small_objects
from skimage.measure import label


def plot_figure(im: numpy.ndarray, tit: str):
    """
     Displays an image with a title using Matplotlib.

     Parameters:
     - im: numpy.ndarray
         The image to be displayed.
     - tit: str
         The title of the image.
     """
    plt.figure(figsize=(10, 5))

    # Display the original image in the first subplot
    plt.imshow(cv2.cvtColor(im,
                            cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for correct display in matplotlib
    plt.title(tit)
    plt.axis('off')  # Hide the axis

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_pair(im1: numpy.ndarray, im2: numpy.ndarray, tit1: str, tit2: str):
    """
       Displays two images side by side with titles.

       Parameters:
       - im1: numpy.ndarray
           The first image (in color).
       - im2: numpy.ndarray
           The second image (in grayscale).
       - tit1: str
           Title for the first image.
       - tit2: str
           Title for the second image.
       """
    plt.figure(figsize=(10, 5))

    # Display the original image in the first subplot
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for correct display in matplotlib
    plt.title(tit1)
    plt.axis('off')  # Hide the axis

    # Display the grayscale image in the second subplot
    plt.subplot(1, 2, 2)
    plt.imshow(im2, cmap='gray')  # Display in grayscale
    plt.title(tit2)
    plt.axis('off')  # Hide the axis

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_hist(title: str, im: numpy.ndarray, pair=False):
    """
       Displays the histogram of an image, optionally alongside the original image.

       Parameters:
       - title: str
           The title for the image.
       - img: numpy.ndarray
           The image for which the histogram will be calculated.
       - pair: bool, optional
           Whether to display the image alongside the histogram (default is False).
       """
    # Calculate the histogram
    histogram = cv2.calcHist([im], [0], None, [256], [0, 255])
    tit = f"{title} Histogram (min {im.min()} - max {im.max()})"
    if pair:
        # Create a figure with 1 row and 2 columns
        plt.figure(figsize=(10, 5))

        # Display the original image in the first subplot
        plt.subplot(1, 2, 1)
        plt.imshow(im, cmap='gray')  # Convert from BGR to RGB for correct display in matplotlib
        plt.title(title)
        plt.axis('off')  # Hide the axis

        # Display the grayscale image in the second subplot
        plt.subplot(1, 2, 2)
        plt.plot(histogram, color='black')
        plt.title(tit)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        # Show the plot
        plt.tight_layout()
        plt.show()
    else:
        # Plot the original image, grayscale image, and histogram
        plt.figure(figsize=(12, 6))
        plt.plot(histogram, color='black')
        plt.title(tit)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        # Show the plot
        plt.tight_layout()
        plt.show()


def plot_grid(images: list[numpy.ndarray], titles: list[str], grid_size=(2, 4), image_size=(300, 300)):
    """
    Displays a grid of images with titles.

    Parameters:
    - images: list of numpy.ndarray
        The list of images to be displayed.
    - titles: list of str
        The list of titles corresponding to the images.
    - grid_size: tuple, optional
        Number of rows and columns in the grid (default is (2, 4)).
    - image_size: tuple, optional
        The size to resize each image (default is (300, 300)).
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1] * 3, grid_size[0] * 3))

    # Flatten axes for easier indexing
    axes = axes.flatten()

    for idx, (img, title) in enumerate(zip(images, titles)):
        if idx >= len(axes):
            print("More images than grid cells, some images won't be shown.")
            break

        # Resize the image to fit into the grid size
        img_resized = cv2.resize(img, image_size)

        # Display the image on the corresponding subplot
        ax = axes[idx]
        ax.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB
        ax.set_title(title)  # Set the image title as the file name
        ax.axis('off')  # Hide axis

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def rotate_and_modify(image: numpy.ndarray, percentage: float, angle: int) -> numpy.ndarray:
    """
    Rotates the image by the specified angle and modifies it by blacking out a portion of the image.

    Parameters:
    - image: numpy.ndarray
        The image to be rotated and modified.
    - percentage: float
        The percentage of the image to modify starting from the first row with white pixels.
    - angle: int
        The angle by which the image should be rotated.

    Returns:
    - rotated_back_image: numpy.ndarray
        The modified image with the original orientation.
    """

    # Get the dimensions of the image
    (h, w) = image.shape[:2]

    # Define the center and the rotation angle
    center = (w // 2, h // 2)
    scale = 1.0

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Find the first row with white pixels (assuming white = 255 for binary image)
    white_pixel_row = None
    for row in range(rotated_image.shape[0]):
        if 255 in rotated_image[row, :]:
            white_pixel_row = row
            break

    # Set all rows from `white_pixel_row` to the calculated percentage to black (if found)
    if white_pixel_row is not None:
        x = white_pixel_row + int(percentage * (h - white_pixel_row))
        rotated_image[white_pixel_row:x, :] = 0

    # Rotate the image back to the original position
    reverse_rotation_matrix = cv2.getRotationMatrix2D(center, -angle, scale)

    # Perform the inverse rotation
    rotated_back_image = cv2.warpAffine(rotated_image, reverse_rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_back_image


def read_images(path: str) -> tuple[list[numpy.ndarray], list[str]]:
    """
       Reads all PNG images from a specified directory and returns them along with their filenames.

       Parameters:
       - path: str
           The directory path where the images are located.

       Returns:
       - images: list of numpy.ndarray
           List of images read from the directory.
       - images_names: list of str
           List of filenames corresponding to the images.
       """
    images = []
    images_names = []
    files = os.listdir(path)
    for file in files:
        # read all .png files
        if file.endswith(".png"):
            images.append(cv2.imread(os.path.join(path, file)))
            images_names.append(file)
    return images, images_names


imgs, imgs_names = read_images("Images")
detected_images = []
for i, img in enumerate(imgs):
    # RGB to grayscale conversion
    im_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plot_pair(img, im_bw, "Original image", "Grayscale image")

    # Pixel intensity analysis via histogram
    plot_hist("Grayscale image", im_bw, True)

    # Image equalization
    equ = cv2.equalizeHist(im_bw)
    plot_hist("Equalized image", equ, True)

    # Image smoothing (noise reduction)
    i_filtered = cv2.GaussianBlur(equ, (7, 7), 1.5)

    plot_pair(equ, i_filtered, "Equalized image", "Filtered image")
    # Image binarization
    i_bin = np.uint8((i_filtered > 135) * 255)
    plot_pair(i_filtered, i_bin, "Filtered image", "Binary image")

    # Label connected regions in the binary image
    objects = label(i_bin)
    # Remove small objects
    large_objects = remove_small_objects(objects, min_size=6000)
    # Convert to uint8 for OpenCV display
    large_objects = np.uint8(large_objects > 0) * 255
    plot_pair(i_bin, large_objects, "Binary image", "Smaller areas removed")

    # Example kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    # Closing operation
    i_close = cv2.morphologyEx(large_objects, cv2.MORPH_CLOSE, kernel)
    plot_pair(large_objects, i_close, "Smaller areas removed", "Closed image")

    # Rotate the image and black-out the first x% rows
    rotated_image = rotate_and_modify(i_close, 0.40, 18)
    plot_pair(i_close, rotated_image, "Closed image", "Blackout image")

    h, w = i_close.shape[:]
    circles = cv2.HoughCircles(
        rotated_image,
        cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=int(w / 5),
        param1=50,
        param2=30,
        minRadius=int(w / 8),
        maxRadius=int(w / 7),
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Round circle parameters
        for i in circles[0, :]:
            # Convert x, y, r to signed integers to prevent overflow
            x, y, r = int(i[0]), int(i[1]), int(i[2])

            # Check if the circle is entirely within image dimensions
            if (x - r >= 0 and y - r >= 0 and  # Top-left of the circle within bounds
                    x + r < img.shape[1] and y + r < img.shape[0]):  # Bottom-right within bounds

                # Draw the outer circle
                cv2.circle(img, (x, y), r, (0, 255, 0), 4)
                # Draw the center of the circle
                cv2.circle(img, (x, y), 4, (0, 0, 255), 4)
    detected_images.append(img)

plot_grid(detected_images[1:], imgs_names[1:], grid_size=(2, 3), image_size=(400, 400))
plot_figure(detected_images[0], imgs_names[0])
