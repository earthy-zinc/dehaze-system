import logging

def create_model(opt):
	model = None
	if opt.model == 'vit':
		print('the model is vit')
		from .mgvit_model import MGVIT
		model = MGVIT()
	else:
		raise NotImplementedError('model [%s] not implemented.' % opt.model)
	model.initialize(opt)
	logging.info("model [%s] was created" % (model.name()))
	return model
