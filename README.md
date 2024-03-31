# Development of document stamp and signature expertise system


1. `extract.py`: Contains functions for extracting stamps and signatures from document images.
2. `separate.py`: Implements algorithms to separate stamps from signatures in overlapped images.
3. `interface.py`: Provides a graphical user interface (GUI) for interacting with the system.


## Signature Verification Dataset
The Signature Verification Dataset used in Methods for signature verification notebook can be found on [Kaggle](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset/data).



## Usage

To use the system:

1. Run `interface.py` to open the GUI.
3. Load an image using the "Load Image" button.
4. Use the "Detect Automatically" button to automatically detect stamps and signatures in the loaded image.
5. Use the "Separate Stamps and Signature" button to separate stamps and signatures.

## Dependencies

This project relies on the following libraries:

- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib.pyplot`)
- scikit-image (`skimage.measure`)
- PyQt5 (`PyQt5`)


