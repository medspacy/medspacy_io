import spacy
from spacy.tokens import Doc, Span, Token


class DocConsumer(object):
	"""A DocConsumer object will consume a spacy doc and output rows based on a configuration provided by the user."""
	def __init__(self, span_type='ent', labels=None, attrs=None, id_attr=None):
		""" Create a DocConsumer object

		Args:
			span_type: the type item to output. Can be {"doc", "ent", "token"}. Default is "ent".
			labels: if outputting entities, output entities with the specified .label_ properties.
				Default None, will output all entites.
			attrs: output the following attributes belonging to span_type. Default None, will output all attrs.
			id_attr: if the Doc object has an identifier that will be output with every row. Default None.
		"""
		super(DocConsumer, self).__init__()
		self.type_dict = {
			"doc":self.get_rows_doc,
			"ent":self.get_rows_ent,
			"token":self.get_rows_token,
			"rels":self.get_rows_rels # for outputting relations at future date
		}

		if span_type in ["doc", "ent", "token"]:
			self.span_type = span_type
		else:
			raise ValueError("Must provide a valid span_type: 'doc', 'ent' or 'token' not {0}".format(span_type))

		if labels:
			self.labels = set(labels)
		else:
			self.labels = None

		if attrs is None: # if no attributes specified, output all
			if span_type is 'ent':
				self.attrs = {"text", "label_", "start", "end"}
			elif span_type is 'token':
				self.attrs = {"text", "like_num", "is_punct"}
			else:
				self.attrs = {"text"}
		else: # otherwise output the specified attributes
			self.attrs = attrs

	def get_rows(self, doc):
		"""Get the rows of the specified document based on object's configs."""
		return self.type_dict[self.span_type](doc)

	def get_rows_doc(self, doc):
		row = []
		for attr in self.attrs:
			val = None
			try:
				val = getattr(doc,attr)
			except AttributeError:
				val = doc._.get(attr)
				#TODO: what happens when there's an exception here?
			row.append(val)
		return row


	def get_rows_ent(self, doc):
		rows = []
		for ent in doc.ents:
			row = []
			if not self.labels or ent.label_ in self.labels:
				for attr in self.attrs:
					val = None
					try:
						val = getattr(ent,attr)
					except AttributeError:
						val = ent._.get(attr)
						#TODO: what happens when there's an exception here?
					row.append(val)
				rows.append(row)
		return rows

	def get_rows_token(self, doc):
		rows = []
		for token in doc:
			row = []
			for attr in self.attrs:
				val = None
				try:
					val = getattr(token,attr)
				except AttributeError:
					val = token._.get(attr)
					#TODO: what happens when there's an exception here?
				row.append(val)
			rows.append(row)
		return rows

	def get_rows_rels(self, doc):
		raise NotImplementedError

def print_rows(rows):
	for row in rows:
		print(row)


