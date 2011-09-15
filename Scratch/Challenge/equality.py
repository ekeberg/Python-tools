import re

f = open('equality.txt')
lines = f.readlines()
#lines = lines[:10]
text = ''.join([l[:-1] for l in lines])


p = re.compile('([a-z][A-Z]{3}[a-z][A-Z]{3}[a-z])')
m = p.findall(text)
