import random
import pandas as pd

def split_rand(data,feature_col,target_col):
	"""data here is a column (feature) and the category. Returns a splitting point and the location of categories.
	Exp:
	{
	'split':3.65,
	'below':0,
	'above':1
	}
	"""
	xsplit = random.uniform(min(data[feature_col]),max(data[feature_col]))
	target_values = set(data[target_col])
	below = target_values.pop()
	above = target_values.pop()
	return {'split':xsplit,'below':below,'above':above}

def find_feature(data,target_col):
	"""returns feature to split on (column name)"""
	columns = [x for x in data.columns if x!=target_col]
	target_values = list(set(data[target_col]))
	assert len(target_values)==2, "number of target values is not 2"
	max_diff = 0
	best_col = columns[0]
	for col in columns:
		data_temp = data.copy()
		amplitude = max(data_temp[col]) - min(data_temp[col])
		data_temp[col] = (data_temp[col] - min(data_temp[col]))/amplitude
		data_temp0 = data_temp[data_temp[target_col]==target_values[0]]
		data_temp1 = data_temp[data_temp[target_col]==target_values[1]]
		avg0 = data_temp0[col].mean()
		avg1 = data_temp1[col].mean()
		diff = abs(avg1 - avg0)
		if diff>max_diff:
			max_diff = diff
			best_col = col
	return best_col

def find_feature_rand(data,target_col):
	columns = [x for x in data.columns if x!=target_col]
	return random.choice(list(columns))

def split(data,feature_col,target_col):
	target_values = list(set(data[target_col]))
	assert len(target_values)==2, "number of target values is not 2"
	data0 = data[data[target_col]==target_values[0]]
	data1 = data[data[target_col]==target_values[1]]
	avg0 = data0[feature_col].mean()
	avg1 = data1[feature_col].mean()
	xsplit = (avg0+avg1)/2.
	below = target_values[0] if avg0<=avg1 else target_values[1]
	above = target_values[0] if avg0>avg1 else target_values[1]
	return {'split':xsplit,'below':below,'above':above}

DEFAULT_PARAMS = {
	'functions':{
		'split':split,
		'find_feature':find_feature
	},
	'names':{
		'target_column':'Target'
	}
}


class tree(object):
	def __init__(self,params = {}):
		self.params = params if params else DEFAULT_PARAMS
		self.root = node(self.params)
	def grow(self,data):
		self.root.grow_rec(data)
	def run(self,data):
		classification = []
		for row in data.iterrows():
			classification.append(self._run_point(row[1]))
		data_new = data.copy()
		data_new[self.params['names']['target_column']] = classification
		return data_new
	def _run_point(self,row):
		return self.root.run_point(row)


class node(object):
	def __init__(self,params,father=None,apriori=None):
		self.params = params
		self.split = self.params.get('functions').get('split')
		self.find_feature = self.params.get('functions').get('find_feature')
		self.father = father
		self.grown = False
		self.apriori = None
		self.endpoint = False
		self.result = None
		self.target_col = self.params.get('names').get('target_column')
	def grow(self,data):
		self.split_feature = self.find_feature(data,self.target_col)
		self.split_manifold = self.split(data[[self.split_feature,self.target_col]],self.split_feature,self.target_col)
		self.grown = True
		return self._split_data(data,self.split_feature,self.split_manifold)
	def _split_data(self,data,split_feature,split_manifold):
		new_data = {}
		new_data[split_manifold['below']] = data[data[split_feature]<=split_manifold['split']]
		new_data[split_manifold['above']] = data[data[split_feature]>split_manifold['split']]
		return new_data
	def grow_rec(self,data):
		if len(set(data[self.target_col]))==1:
			self.endpoint = True
			self.result = set(data[self.target_col]).pop()
			self.grown = True
			return
		new_data = self.grow(data)
		self.below = node(self.params,father=self,apriori=self.split_manifold['below'])
		self.above = node(self.params,father=self,apriori=self.split_manifold['above'])
		self.below.grow_rec(new_data[self.split_manifold['below']])
		self.above.grow_rec(new_data[self.split_manifold['above']])
	def run_point(self,row):
		assert self.grown, "Node isn't grown"
		if self.endpoint: return self.result
		if row[self.split_feature]<=self.split_manifold['split']:
			return self.below.run_point(row)
		if row[self.split_feature]>self.split_manifold['split']:
			return self.above.run_point(row)
		assert False, "Shouldn't reach this point"

df = pd.DataFrame({'Target':[0,1,1,1,0,0,1,0,1,0],'x':[-5.,-2.3,6.1,2.1,2.5,2.5,0.,0.2,9.,11.5],'y':[152,89,56,54,128,1045,85,3,8,523],'z':[-89.9,-85.,-56.,-125.,-547.,-20.,-20.,-96.,-65.8,-2.],'t':[5,-89,-56,-2,56,1,-215,28,5,47]})

t = tree()

t.grow(df)

print t.root.split_feature
print t.root.split_manifold

print t.run(df[[x for x in df.columns if x != 'Target']])


dftest = pd.DataFrame({'x':[0.,2.],'y':[8,9],'z':[-100.,-85.],'t':[-30.,45]})

print t.run(dftest)