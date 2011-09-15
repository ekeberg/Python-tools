import zipfile
import re

f = zipfile.ZipFile('channel.zip','r')

start_file = '90052'

p = re.compile('nothing is ([0-9]+)')

info_list = []

next_file = start_file

count = 0
while True:
    count += 1
    info = f.getinfo(next_file+'.txt')
    info_list.append(info.comment)
    data = f.read(next_file+'.txt')
    m = p.search(data)
    next_file = m.groups()[0]
