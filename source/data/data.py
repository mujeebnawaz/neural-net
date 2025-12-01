from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlparse
from urllib.request import urlopen
import numpy as np

class Data:

	data_location = './data/datasets'

	def __init__(self, source):
		self.source = source

		file_name = self.source.path.split('/')[-1]
		self.file = Data.data_location+'/'+file_name

	"""
	Method to download the data. 
	"""
	def download(self):
		# Get the instance of connection based on the source's scheme. 
		if self.source.scheme == 'https':
			connection = HTTPSConnection(self.source.netloc)
		else:
			connection = HTTPConnection(self.source.netloc)

		connection.request('GET', self.source.path)
		response = connection.getresponse()

		if response.status != 200:
			connection.close()
			raise Exception("Invalid URL")
		
		if response.getheader('Content-Type').startswith('application/'):
			self.store_data(response.read(),self.file)

		connection.close()
	
	@staticmethod
	def store_data(data, file):
		with open(file, 'wb') as f:
			f.write(data)

	@staticmethod
	def store_numpy(data, file, multiple=False):
		if not multiple:	
			np.save(file, data)
		else:
			np.savez(file, data)

		print(f"Array saved to '{file}' with shape {data.shape} and dtype {data.dtype}.")


		...

	def validate_source(self, url):
		try:
			with urlopen(url) as response:
				return True
		except Exception as e:
			return False

	@property
	def source(self):
		return self._source
	
	@source.setter
	def source(self, source):
		if not source: 
			raise ValueError("A URL is required.")
		if not self.validate_source(source):
			raise ValueError("Invalid URL.")

		# Parse URL and store it in the object. 
		self._source = urlparse(source)

	def __str__(self):
		return f"Source: {self.source}"



