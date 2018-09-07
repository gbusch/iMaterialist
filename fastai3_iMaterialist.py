# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = "data/"
sz = 299
bs = 18

arch=resnext50
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=8, test_name='test')
learn = ConvLearner.pretrained(arch, data, precompute=False, ps=0.5)

#learn.load('rx50_save2_precomputeFalse.h5')
learn.unfreeze()
lr=np.array([5e-5,5e-4,2e-3])
#learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.load('rx50_save3_differential.h5')

learn.fit(lr, 3, cycle_len=1, cycle_mult=2)

print('predictions for valid set')
log_preds,y = learn.TTA(is_test=True)
probs = np.mean(np.exp(log_preds),0)
preds = np.argmax(probs, axis=1)

realpred = pd.Series(preds).map(lambda x: data.classes[x])
testnames = data.test_ds.fnames

subm = pd.DataFrame()
subm['id'] = pd.Series(testnames).map(lambda x: x[5:-4]).astype(int)
subm['predicted'] = pd.Series(realpred).astype(int)

subm.set_index('id', inplace=True)
new_index = pd.Index(np.arange(1,12801,1), name="id")
subm = subm.reindex(new_index)

subm = subm.fillna(0)
subm = subm.astype(int)
subm.to_csv('subm_full_rx50_2.csv')