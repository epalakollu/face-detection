from PIL import Image
import glob

prefix = 's41'
counter = 0
for filename in glob.glob('images/data/s41/*.pgm'):
  im = Image.open(filename)
  counter += 1
  print(prefix+str(counter)+'.jpeg')
  im.save(prefix+str(counter)+'.jpeg')