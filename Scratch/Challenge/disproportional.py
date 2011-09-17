import xmlrpclib

names = ['evil', 'phonebook', 'evil1', 'evil2', 'evil3']

base_url = 'http://www.pythonchallenge.com/pc/'
urls = [base_url+n+'.php' for n in names]

def try_adresses(names, urls):
    for url,n in zip(urls,names):
        try:
            server = xmlrpclib.Server(url)
            print "%s: %s" % (n, server.phone('5'))
        except xmlrpclib.ProtocolError:
            print "%s doesn't work" % n


def try_arguments(adress, arguments):
    server = xmlrpclib.Server(adress)
    for a in arguments:
        print "%s: %s" % (a, server.phone(a))
    return server

arguments = ['Bert','bert','BERT']
server = try_arguments('http://www.pythonchallenge.com/pc/phonebook.php',arguments)

phone_number = '555-48259' #'555-ITALY'
