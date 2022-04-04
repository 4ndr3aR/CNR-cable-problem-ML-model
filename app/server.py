#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd

import torch
import aiohttp
import asyncio
import uvicorn

from fastai import *
#from fastai.vision import *
from fastai.tabular import *
from fastai.tabular.learner import TabularLearner
from fastai.learner import load_learner
from pathlib import Path

import hashlib

from io import BytesIO
from io import StringIO

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from flask import Flask, request
from flask_debugtoolbar import DebugToolbarExtension
import logging

from wwf.tab.export import *

from threading import Thread

import kistlerfile
from kistlerfile import KistlerFile
from kistlerfile import create_inference_ready_sample

import argparse

from argument_parser import define_boolean_argument
from argument_parser import var2opt


#export_file_url  = 'http://deeplearning.ge.imati.cnr.it/ditac/models/ditac-cable-problem-v0.6-endoftraining.pt'
#export_to_url    = 'http://deeplearning.ge.imati.cnr.it/ditac/models/ditac-cable-problem-v0.6-endoftraining-TabularPandas-object-to.pkl'
#export_df_url    = 'http://deeplearning.ge.imati.cnr.it/ditac/models/ditac-cable-problem-v0.6-endoftraining-df.csv'
#export_pkl_url   = 'http://deeplearning.ge.imati.cnr.it/ditac/models/ditac-cable-problem-v0.6-endoftraining.pkl'
#export_file_name = 'ditac-cable-problem-v0.6-endoftraining.pt'
#export_to_name   = 'ditac-cable-problem-v0.6-endoftraining-TabularPandas-object-to.pkl'
#export_df_name   = 'ditac-cable-problem-v0.6-endoftraining-df.csv'
#export_pkl_name  = 'ditac-cable-problem-v0.6-endoftraining.pkl'

classes = [False, True]
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner(url, model_name):
	#await download_file(export_file_url, path / export_file_name)
	#await download_file(export_to_url, path / export_to_name)
	#await download_file(export_df_url, path / export_df_name)
	if url is None or model_name is None:
			message = "\n\nNo model download URL or model destination name has been specified. Exiting...\n"
			raise RuntimeError(message)
	print(f'Downloading model from: {url} with model name: {model_name}')
	await download_file(url, path / model_name)
	try:
		'''
		to      = load_pandas(path / export_to_name)
		print(f'{to = }')

		df = pd.read_csv(path / export_df_name).T
		df = df.astype({col: np.float16 for col in df.columns[:-1]})
		df = df.convert_dtypes()
		to_new = to.train.new(df)
		print(f'{to_new = }')
		to_new.process()
		dls_new = to_new.dataloaders(bs=8)

		model   = torch.load(f'{path}/{export_file_name}')
		print(f'{model = }')
		learn   = TabularLearner(dls_new, model, to.loss_func)
		'''
		learn = load_learner(path / model_name)
		print(f'{learn = }')
		#learn = load_learner(path, export_file_name)
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
	#row.show()
	#probs
	return row, clas, probs


# --------------------------------------------------
# ====================== HTML ======================
# --------------------------------------------------
@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
	form_data = await request.form()
	starlette_data = await(form_data['file'].read())
	csvdata = StringIO(str(starlette_data.decode("utf-8")))

	row, clas, probs = process_csvdata(csvdata)

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

@flask_app.route('/post', methods=['POST'])
def result():
	#print(request.form['kistlerfile'])
	print(f'\n\nReceived filename: {Path(request.form["filename"]).stem}\n\n')
	content = request.form['kistlerfile']
	#return f"Received: {request.form['filename']} -> {request.form['kistlerfile']}"
	readable_hash = hashlib.sha256(content.encode('utf-8')).hexdigest();

	csvdata = StringIO(str(content))

	row, clas, probs = process_csvdata(csvdata)
	print(f'\n\nPrediction for filename: {Path(request.form["filename"]).stem} -> class: {str(classes[int(clas)])} -> probs: {str(probs)}\n\n')

	return f"Received: {request.form['filename']} -> sha256: {readable_hash} -> class: {str(classes[int(clas)])} -> probs: {str(probs)}"
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

	'''
export_pkl_url   = 'http://deeplearning.ge.imati.cnr.it/ditac/models/ditac-cable-problem-v0.6-endoftraining.pkl'
export_pkl_name  = 'ditac-cable-problem-v0.6-endoftraining.pkl'
	'''

	parser.add_argument('--cmd',		default="serve"		, help='the function to execute, default: serve')
	parser.add_argument('--model-name'				, help='the model to load for inference in .pkl format')
	parser.add_argument('--model-url'				, help='the URL where to download the model')
	parser.add_argument('--web-port',	default=55564, type=int	, help='web interface (for debug purposes) port')
	parser.add_argument('--flask-port',	default=55563, type=int	, help='flask TCP port where to receive POST requests')

	args = parser.parse_args()

	print(f'argument_parser() received arguments: {args}')

	return args


args = argument_parser()

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner(args.model_url, args.model_name))]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

if __name__ == '__main__':
	if 'serve' in args.cmd:
		print(f'Starting main python script: {__name__}...')

		host='0.0.0.0'
		#port=55513
		#port=55563
		port=args.flask_port
		check_port(port)
		#start_flask(port, host, debug=flask_debug)
		t = Thread(target=start_flask, args=(port, host, flask_debug,))
		t.start()

		host='0.0.0.0'
		#port=55514
		#port=55564
		port=args.web_port
		check_port(port)
		print(f'Creating Uvicorn app with {host = }, {port = }')
		uvicorn.run(app=app, host=host, port=port, log_level="info")			# HTML interface



print(f'Main python script: {__name__} reached the end...')

