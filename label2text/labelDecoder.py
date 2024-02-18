from PIL import Image
import pytesseract

image = Image.open('label2.jpeg')
text = pytesseract.image_to_string(image)
print(text)
# Ingrediente: Amidon din porumb, grésime de palmier, faina de porumb, zahar, faina de sola,