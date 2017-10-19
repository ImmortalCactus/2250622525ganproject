import urllib.request  as urllib
import os
from PIL.Image import core as Image
number=0
with open("imagenet.synset.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 
os.chdir('./file')
for i in content:
    number=number+1
    name=str(str(number)+".jpg")
    try:    
        url=str(i)
        image=urllib.URLopener()
        image.retrieve(url,name)
        im = open(name)
        im.save(name, dpi=(600,600))
    except IOError:
        continue
