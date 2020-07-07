import torch


def safe_cuda_or_cpu(th_object):
	try:
		return th_object.cuda()
	except:
		return th_object.cpu()
