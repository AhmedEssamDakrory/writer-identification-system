import os
import xml.etree.ElementTree as ET

from constants import INPUTS_DIR

arr = os.listdir('xml')
f = open(INPUTS_DIR + '/writer-id.txt', "w")
for file in arr:
    root = ET.parse('xml/' + str(file)).getroot()
    f.write(str(file)+':'+str(root.attrib['writer-id']+'\n'))
f.close()