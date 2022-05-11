#!/usr/bin/env python3

import bcrypt

def encrpyt_password(plaintext_password, salt_rounds=12, debug=False):
	salt = bcrypt.gensalt(salt_rounds)
	if debug:
		print(f'bcrypt generated salt: {salt}')
	hashed = bcrypt.hashpw(plaintext_password.encode('UTF8'), salt)
	if debug:
		print(f'bcrypt salted and hashed password: {hashed}')
	return hashed, salt

def verify_password(plaintext_password, hashed_password, salt=b'', debug=False):
	if debug:
		retrieved_salt = hashed.find(salt)
		print(f'Correct salt from bcrypt hashed password? {retrieved_salt}')
	hashed2 = bcrypt.hashpw(plaintext_password.encode('UTF8'), hashed)
	if debug:
		print(f'bcrypt salted and hashed password (to be verified): {hashed2}')
	match = (hashed == hashed2)
	if debug:
		print(f'The two salted and hashed passwords match? {match}')
	return hashed2, match

if __name__ == '__main__':

	debug=True

	hashed, salt = encrpyt_password('I love DigitBrain', debug=debug)
	verify_password(plaintext_password='I love DigitBrain', hashed_password=hashed, debug=debug)
