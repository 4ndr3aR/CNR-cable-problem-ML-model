#!/usr/bin/env python

import os
import sys
import time
import signal

import numpy as np
import pandas as pd

import torch
import aiohttp
import asyncio
import uvicorn

from fastai import *
from fastai.tabular import *
from fastai.tabular.learner import TabularLearner
from fastai.learner import load_learner
from pathlib import Path

import hashlib

import shutil

from io import BytesIO
from io import StringIO

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from starlette.authentication import AuthCredentials, AuthenticationBackend, AuthenticationError, SimpleUser, requires, UnauthenticatedUser
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route


from starlette.endpoints import HTTPEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocketDisconnect



import base64
import binascii

from flask import Flask, request
from flask_debugtoolbar import DebugToolbarExtension
import logging

from wwf.tab.export import *

from threading import Thread

from colored import fg, bg, attr

import kistlerfile
from kistlerfile import KistlerFile
from kistlerfile import create_inference_ready_sample

import argparse

from argument_parser import define_boolean_argument
from argument_parser import var2opt

from bcrypt_password import encrpyt_password, verify_password



rst = attr("reset")				# just to colorize text
classes = [False, True]
path = Path(__file__).parent
user_pass_db = None				# to be populated later in main...


class BasicAuthBackend(AuthenticationBackend):
	async def authenticate(self, conn):
		if "Authorization" not in conn.headers:
			return

		auth = conn.headers["Authorization"]
		try:
			scheme, credentials = auth.split()
			if scheme.lower() != 'basic':
				return
			decoded = base64.b64decode(credentials).decode("ascii")
		except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
			raise AuthenticationError('Invalid basic auth credentials')

		username, _, password = decoded.partition(":")
		# TODO: You'd want to verify the username and password here.
		print(f'{username = } - {password = }')
		password_verified, msg = verify_user_and_password(username, password)
		if not password_verified:
			#return f"BasicAuthBackend received username: {username} -> {msg}{username}"
			return AuthCredentials(), UnauthenticatedUser()
		return AuthCredentials(["authenticated"]), SimpleUser(username)

@requires("authenticated", redirect="login")
async def homepage(request):
	if request.user.is_authenticated:
		html_file = path / 'view' / 'index.html'
		#return PlainTextResponse('Hello, ' + request.user.display_name)
		return HTMLResponse(html_file.open().read())
	return PlainTextResponse('Hello, unauthenticated user!')

@requires("authenticated", redirect="login")
async def homepage(request):
    print(f'Requested home page via @app.route("/")')
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())







# --------------------------------------------------
# ============== ROUTES AND ENDPOINTS ==============
# --------------------------------------------------
@requires("authenticated", redirect="homepage")
async def admin(request):
	return JSONResponse(
	{
		"authenticated": request.user.is_authenticated,
		"user": request.user.display_name,
	}
	)
# --------------------------------------------------
# ==================================================
# --------------------------------------------------








routes = [
		Route("/admin", endpoint=admin),
		Route("/test", endpoint=homepage),
		Route("/", endpoint=homepage),
		#Route("/login", endpoint=login),
	]

middleware = [Middleware(AuthenticationMiddleware, backend=BasicAuthBackend())]

#DEBUG = config('DEBUG', cast=bool, default=False)

app = Starlette(routes=routes, middleware=middleware) #, debug=DEBUG)

#app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner(cmd, url, model_name):
	if 'serve' in cmd:
		if url is None or model_name is None:
				message = "\n\nNo model download URL or model destination name has been specified. Exiting...\n"
				raise RuntimeError(message)
	else:
		# do nothing, we're being called by docker's RUN instruction
		return
	print(f'Downloading model from: {url} with model name: {model_name}')
	await download_file(url, path / model_name)
	try:
		learn = load_learner(path / model_name)
		print(f'{learn = }')
		return learn
	except RuntimeError as e:
		if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
			print(e)
			message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
			raise RuntimeError(message)
		else:
			raise



def process_csvdata(csvdata):
	debug = False
	if debug:
		df = pd.read_csv(csvdata, skiprows=KistlerFile.kistler_dataframe_line_offset, sep=';', decimal=',')
		print(f'{df = }')
	kf = KistlerFile(fname=csvdata,debug=False)		# let's see if this trick works...
	if debug:
		prediction = kf.df

	KistlerFile._kf_max_time = 3.9999			# this is mandatory and hardcoded to obtain (exactly) 800 samples after resampling. ML model expects 800 columns.
	kf.resample(debug=False)

	sample = create_inference_ready_sample(kf, debug=False)
	row, clas, probs = learn.predict(sample)
	return row, clas, probs, kf


# --------------------------------------------------
# ====================== HTML ======================
# --------------------------------------------------
@app.route("/login")
async def login(request):
	if request.user.is_authenticated:
		return RedirectResponse(url=f"/", status_code=303)
	else:
		response = Response(headers={"WWW-Authenticate": "Basic"}, status_code=401)
	return response

'''
@requires("authenticated", redirect="login")
@app.route('/')
async def homepage(request):
    print(f'Requested home page via @app.route("/")')
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())
'''

@app.route('/analyze', methods=['POST'])
async def analyze(request):
	form_data = await request.form()
	starlette_data = await(form_data['file'].read())
	csvdata = StringIO(str(starlette_data.decode("utf-8")))

	row, clas, probs, kf = process_csvdata(csvdata)

	return JSONResponse({'result': str(classes[int(clas)]) + ' -> ' + str(probs)})
# --------------------------------------------------
# ==================================================
# --------------------------------------------------

flask_debug = False
toolbar     = None
flask_app   = Flask(__name__)

# --------------------------------------------------
# ====================== POST ======================
# --------------------------------------------------
@flask_app.route('/debug')
def index():
    logging.warning("See this message in Flask Debug Toolbar!")
    return "<html><body>it works, now try a post request...</body></html>"

def verify_user_and_password(username, plaintext_password, debug=False):
	print(f'\n\nReceived username: {fg("chartreuse_2a")}{username}{rst} and password: {fg("turquoise_2")}{plaintext_password}{rst}\n\n')
	if username in user_pass_db:
		db_password = user_pass_db[username]
	else:
		msg = 'Authentication denied, no such user: '
		print(f'{fg("red_1")}{36*"-"}{rst}')
		print(f'{fg("red_1")}{msg}{rst}{fg("chartreuse_2a")}{username}{rst}')
		print(f'{fg("red_1")}{36*"-"}{rst}')
		print('\n')
		#return f"/post received username: {username} -> {msg}{username}"
		return False, msg

	hashed,  salt  = encrpyt_password(plaintext_password, debug=debug)
	hashed2, match = verify_password(plaintext_password=db_password, hashed_password=hashed, debug=debug)

	if match:
		print(f'{fg("chartreuse_2a")}{22*"-"}{rst}')
		print(f'{fg("chartreuse_2a")}Authentication granted{rst}')
		print(f'{fg("chartreuse_2a")}{22*"-"}{rst}')
		print('\n')
		return True, ''
	else:
		msg = 'Authentication denied, passwords do not match for: '
		print(f'{fg("red_1")}{45*"-"}{rst}')
		print(f'{fg("red_1")}{msg}{rst}')
		print(f'{fg("red_1")}{45*"-"}{rst}')
		print('\n')
		return False, msg

@flask_app.route('/post', methods=['POST'])
def result():
	debug = True
	if 'username' in request.form and 'password' in request.form:
		username            = request.form['username']
		plaintext_password  = request.form['password']
	else:
		print(f'\n\n{fg("white")}{bg("red_1")}No username and/or password received, continuing in "compatibility mode" {attr("blink")}(deprecated, will be removed in the future){rst}')
		username            = 'user1'
		plaintext_password  = 'password1'

	password_verified, msg = verify_user_and_password(username, plaintext_password)
	if not password_verified:
		return f"/post received username: {username} -> {msg}{username}"

	print(f'Received filename: {Path(request.form["filename"]).stem}\n\n')
	content = request.form['kistlerfile']
	readable_hash = hashlib.sha256(content.encode('utf-8')).hexdigest();

	csvdata = StringIO(str(content))

	row, clas, probs, kf = process_csvdata(csvdata)
	predstr = str(classes[int(clas)])

	print(f'\n\nPrediction for filename: {Path(request.form["filename"]).stem} -> class: {predstr} -> probs: {str(probs)}\n\n')

	basepath = Path('/app/static')

	graphfn     = basepath / 'graph' / (Path(request.form["filename"]).stem + '.png')
	resampledfn = basepath / 'graph' / (Path(request.form["filename"]).stem + '-resampled.png')
	csvfn       = basepath / 'data'  / (Path(request.form["filename"]).name)
	htmlfn      = basepath / 'html'  / (Path(request.form["filename"]).stem + '.html')

	print(f'\nWriting graph to: {graphfn}, CSV file to {csvfn} and HTML file to {htmlfn}')

	graphfn.parent.mkdir(exist_ok=True, parents=True)
	csvfn.parent.mkdir(exist_ok=True, parents=True)
	htmlfn.parent.mkdir(exist_ok=True, parents=True)

	kf.graph(resampled=False, both=True, debug=False, filename=graphfn)

	with open(csvfn, 'w') as fd:
		csvdata.seek(0)
		shutil.copyfileobj(csvdata, fd)

	graphfnlink	= Path('..') / Path(*graphfn.parts[3:])		# remove leading '/app/static' (because our relative path is deeplearning.ge.imati.cnr.it:55564/static/html)
	csvfnlink	= Path('..') / Path(*csvfn.parts[3:])		# remove leading '/app/static' (because our relative path is deeplearning.ge.imati.cnr.it:55564/static/html)
	print(f'Writing links: {graphfnlink} - {csvfnlink}')
	template    = open(basepath / 'template.html').read().format(csv_fn=csvfnlink, graph_fn=graphfnlink, pred=predstr)

	with open(htmlfn, "w") as html_file:
		html_file.write(template)

	return f"/post received: {request.form['filename']} -> sha256: {readable_hash} -> class: {predstr} -> probs: {str(probs)}"
# --------------------------------------------------
# ==================================================
# --------------------------------------------------

def load_pandas(fname):
	"Load in a `TabularPandas` object from `fname`"
	distrib_barrier()
	res = pickle.load(open(fname, 'rb'))
	return res

def start_flask(port, host='0.0.0.0', debug=False):
	print(f'Running Flask app with {host = }, {port = }')
	if flask_debug:
		app.secret_key = 'asdfasdfqwerqwer'
		toolbar = DebugToolbarExtension(flask_app)
	flask_app.run(host, port, debug)							# POST requests
	return flask_app


def check_port(port):
	if type(port) != int or port < 1 or port > 65535:
		message = "\n\nPlease provide a valid port number. Exiting...\n"
		raise RuntimeError(message)

def argument_parser():
	parser = argparse.ArgumentParser(description='Image Segmentation Inference with Fast.ai v2 and SemTorch')

	parser.add_argument('--cmd',		default=""		, help='the function to execute, default: serve')
	parser.add_argument('--model-name'				, help='the model to load for inference in .pkl format')
	parser.add_argument('--model-url'				, help='the URL where to download the model')
	parser.add_argument('--web-port',	default=55564, type=int	, help='web interface (for debug purposes) port')
	parser.add_argument('--flask-port',	default=55563, type=int	, help='flask TCP port where to receive POST requests')

	args = parser.parse_args()

	print(f'argument_parser() received arguments: {args}')

	return args

def create_user_pass_db():
	user_pass_db = {
				'user1': 'password1',
				'user2': 'password2',
				'user3': 'password3',
				'user4': 'password4',
			}

	external_user = os.environ['EXT_USERNAME']
	external_pass = os.environ['EXT_PASSWORD']
	user_pass_db[external_user] = external_pass
	print(f'user_pass_db: {user_pass_db}')
	return user_pass_db


args = argument_parser()

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner(args.cmd, args.model_url, args.model_name))]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

if __name__ == '__main__':
	if 'serve' in args.cmd:
		print(f'Starting main python script: {__name__}...')

		user_pass_db = create_user_pass_db()

		host='0.0.0.0'
		port=args.flask_port
		check_port(port)
		t = Thread(target=start_flask, args=(port, host, flask_debug,))
		t.start()

		host='0.0.0.0'
		port=args.web_port
		check_port(port)
		print(f'Creating Uvicorn app with {host = }, {port = }')
		uvicorn.run(app=app, host=host, port=port, log_level="info")			# HTML interface
		print(f'Killing self pid: {os.getpid()} to get rid of Flask...')
		time.sleep(2)
		os.kill(os.getpid(), signal.SIGKILL)

print(f'Main python script: {__name__} reached the end...')

