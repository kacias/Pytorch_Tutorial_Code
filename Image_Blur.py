from PIL import Image, ImageFilter

im = Image.open('python.png')
blurImage = im.filter(ImageFilter.BLUR)

blurImage.save('python-blur.png')