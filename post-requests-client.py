#!/usr/bin/env python

import requests
port = 55513
print(f'Performing post request on port {port}\n')
r = requests.post(f'http://deeplearning.ge.imati.cnr.it:{port}/post', data={'kistlerfile': 'fake base64-encoded data'})
# And done.
print(f'Answer:')
print(20*'-')
print(r.text) # displays the result body.
print(20*'-')

