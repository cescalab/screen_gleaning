The scripts and font files to generate the different examples for the experiment are present in `ImageGeneration`.

The scripts are a Bash scripts that compiles images in PDF files with `Xelatex` from a `Tex` template and convert PDFs to JPG format. It is possible to modify the width and height of the image as well as the density per inches (dpi) of the JPG with variables in the script.

- `ImageGeneration/Letter` create an image set of eye doctor letters (C D E F L N O P T Z) in the Sloan font (available at https://github.com/denispelli/Eye-Chart-Fonts.git) with parametrizable sizes. Each image shows one centered letter with the specified font size.
- `ImageGeneration/SMS`creates an image set of SMS-like messages as can be obtained with on an smartphone containing an authentication code. The size of set can be configurable and each image have a randomly generated six-digit code.
- `ImageGeneration/Grid` creates an image set of a grid of digits (40x40). The digits are randomly distributed for each image. 
