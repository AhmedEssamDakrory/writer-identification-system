import os
import xml.etree.ElementTree as ET

arr = os.listdir('/home/mazen/Downloads/xml')
f = open('writer-id.txt', "a")
for file in arr:
    root = ET.parse('/home/mazen/Downloads/xml/' + str(file)).getroot()
    f.write(str(file)+':'+str(root.attrib['writer-id']+'\n'))
f.close()