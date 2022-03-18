#!/usr/bin/env python3.8

import sys
import requests

if len(sys.argv) < 2:
	print(f'Please provide a KistlerFile name to read...')
	sys.exit()

kfname = sys.argv[1]
print(f'Reading filename: {kfname}')

with open(kfname, 'rb') as kfd:
	kfdata = kfd.read()

print(f'{kfdata = }')

port = 55563
print(f'Performing post request on port {port}\n')
r = requests.post(f'http://deeplearning.ge.imati.cnr.it:{port}/post', data={'filename': kfname, 'kistlerfile': kfdata})

# And done.
print(f'Answer:')
print(20*'-')
print(r.text) # displays the result body.
print(20*'-')

