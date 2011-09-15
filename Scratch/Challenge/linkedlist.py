import urllib2
import re

p = re.compile('the next nothing is ([0-9]+)')

base_url = 'http://www.pythonchallenge.com/pc/def/linkedlist.php?nothing='
start = 82682

response = urllib2.urlopen(base_url+str(start))
text = response.read()
texts = []
texts.append(text)

m = p.search(texts[-1])
next_url = m.groups()[0]
while(1):
    response = urllib2.urlopen(base_url+next_url)
    texts.append(response.read())
    m = p.search(texts[-1])
    next_url = m.groups()[0]
    print texts[-1]
