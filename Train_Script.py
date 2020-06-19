from fastai2.vision.all import *
import torch
from fastscript import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from efficientnet_pytorch import EfficientNet
import gc
path = Path('./DDSM_NOBARS/'); path.ls()

def setup_transforms(size=224, normalize=True, aug=True):
    if normalize and aug:
        batch_tfms = [IntToFloatTensor(),
              *aug_transforms(size=224, max_warp=0, min_scale=0.75),
              Normalize.from_stats(*imagenet_stats)]
        item_tfms = [ToTensor(), Resize(460)]
    if normalize and not aug:
        batch_tfms = [IntToFloatTensor(),
              Normalize.from_stats(*imagenet_stats)]
        item_tfms = [ToTensor(), Resize(460)]
    if not normalize and aug:
        batch_tfms = [IntToFloatTensor(),
              *aug_transforms(size=224, max_warp=0, min_scale=0.75),
             ]
        item_tfms = [ToTensor(), Resize(460)]
    if not normalize and not aug:
        batch_tfms = [IntToFloatTensor()]
        item_tfms = [ToTensor(), Resize(460)]  
    return batch_tfms, item_tfms

def get_dsrc(train_imgs, tst_imgs):
    random.shuffle(train_imgs)
    start_val = len(train_imgs) - int(len(train_imgs)*.2) # last 20% validation
    idxs = list(range(start_val, len(train_imgs)))
    splits = IndexSplitter(idxs)
    split = splits(train_imgs)
    split_list = [split[0], split[1]]
    split_list.append(L(range(len(train_imgs), len(train_imgs)+len(tst_imgs))))
    return Datasets(train_imgs+tst_imgs, tfms=[[PILImage.create], [parent_label, Categorize]],
                splits = split_list)

def get_train_labels(dsrc):
    train_labels = L()
    for i in range(len(dsrc.train)):
        train_labels.append(dsrc.train[i][1])
    for i in range(len(dsrc.valid)):
        train_labels.append(dsrc.valid[i][1])
    return train_labels

@call_parse
def main(
        lr:Param("Learning Rate", float)=3e-3,
        size:Param("Image Size", int)=224,
        epochs:Param("Epochs", int)=25,
        bs:Param("Batch Size", int)=64,
        mixup_arg:Param("Mixup?:", bool_arg)=False,
        mixup_alpha:Param("Mixup Alpha ", float)=0.4,
        opt:Param("Optimizer", str)='adam',
        loss: Param("Loss function", str)='CrossEntropy',
        arch:Param("CNN Architecture", str)='resnet34',
        lrfinder:Param("Run learning rate finder; don't train")=0,
        pretrained:Param("Use pre-trained network?", bool_arg)=True,
        normalize:Param("Normalize dataset with ImageNetStats?", bool_arg)=True,
        aug:Param("Use data augmentation?", bool_arg)=True,
        scenario:Param("Scenario Number", str)='1'
        ):

        #Optimizers
        if   opt=='adam': opt_func = Adam
        elif opt=='rms': opt_func = RMSprop
        elif opt=='sgd': opt_func = SGD
        #Models
        if arch == 'resnet18': arch_m = resnet18
        elif arch == 'resnet34': arch_m = resnet34
        elif arch == 'resnet50': arch_m = resnet50
        elif arch == 'xresnet18': arch_m = xresnet18
        elif arch == 'xresnet34': arch_m = xresnet34
        elif arch == 'xresnet50': arch_m = xresnet50
        elif arch == 'densenet121': arch_m = densenet121
        elif arch == 'densenet161': arch_m = densenet161
        elif arch == 'densenet169': arch_m = densenet169
        elif arch == 'densenet201': arch_m = densenet201
        elif arch == 'squeezenet1_0': arch_m = squeezenet1_0
        elif arch == 'squeezenet1_1': arch_m = squeezenet1_1
        elif arch == 'efficientnetb3':
            arch_m = EfficientNet.from_pretrained("efficientnet-b3", advprop=True)
            arch_m._fc = nn.Linear(1536, 2)
        elif arch == 'efficientnetb4':
            arch_m = EfficientNet.from_pretrained("efficientnet-b4", advprop=True)
            arch_m._fc = nn.Linear(1792, 2)    
        elif arch == 'efficientnetb7':
            arch_m = EfficientNet.from_pretrained("efficientnet-b7", advprop=True)
            arch_m._fc = nn.Linear(2560, 2)
        elif arch == 'efficientnetb8':
            arch_m = EfficientNet.from_pretrained("efficientnet-b8", advprop=True)
            arch_m._fc = nn.Linear(2816, 2)
        #TODO add effNets!
        
        #Loss
        if loss == 'CrossEntropy': loss_fn = CrossEntropyLossFlat()
        elif loss == 'LabelSmoothing': loss_fn = LabelSmoothingCrossEntropy()
        #Transformations
        batch_tfms, item_tfms = setup_transforms(size=size, normalize=normalize, aug=aug)
        #Get train and test images
        train_imgs = get_image_files(path/'train')
        tst_imgs = get_image_files(path/'test')
        #Get Dataset
        dsrc = get_dsrc(train_imgs, tst_imgs)
        #Get Train Labels
        train_labels = get_train_labels(dsrc)
        #lrfinder
        if lrfinder:
            dls = dsrc.dataloaders(bs=bs, after_item=item_tfms, after_batch=batch_tfms)
            learn = cnn_learner(dls, arch_m, pretrained=pretrained, normalize = normalize, opt_func = opt_func, metrics=accuracy, loss_func = loss_fn).to_fp16()
            learn.lr_find()
        #CrossVal and Train!
        random.shuffle(train_imgs)
        if not lrfinder:
            run_count = 0
            val_pct = []
            tst_preds = []
            skf = StratifiedKFold(n_splits=10, shuffle=True)
            for _, val_idx in skf.split(np.array(train_imgs), train_labels):
                splits = IndexSplitter(val_idx)
                split = splits(train_imgs)
                split_list = [split[0], split[1]]
                split_list.append(L(range(len(train_imgs), len(train_imgs)+len(tst_imgs))))
                dsrc = Datasets(train_imgs+tst_imgs, tfms=[[PILImage.create], [parent_label, Categorize]],
                                splits=split_list)
                dls = dsrc.dataloaders(bs=bs, after_item=item_tfms, after_batch=batch_tfms)
                if not mixup_arg and "efficientnet" not in arch:
                    learn = cnn_learner(dls, arch_m, pretrained=pretrained, normalize = normalize, opt_func = opt_func, metrics=accuracy, loss_func = loss_fn).to_fp16()
                if mixup_arg and "efficientnet" not in arch:
                    print("Using mixup with an alpha of", mixup_alpha)
                    learn = cnn_learner(dls, arch_m, pretrained=pretrained, normalize = normalize, opt_func = opt_func, metrics=accuracy, loss_func = loss_fn, cbs=MixUp(mixup_alpha)).to_fp16()
                if not mixup_arg and "efficientnet" in arch:
                    learn = Learner(dls, arch_m, normalize = normalize, opt_func = opt_func, metrics=accuracy, loss_func = loss_fn).to_fp16()
                if mixup_arg and "efficientnet" in arch:
                    print("Using mixup with an alpha of", mixup_alpha)
                    learn = Learner(dls, arch_m, normalize = normalize, opt_func = opt_func, metrics=accuracy, loss_func = loss_fn, cbs=MixUp(mixup_alpha)).to_fp16()
                print('-'*20 + "Run #" + str(run_count+1) + '-'*20)
                if not pretrained:
                    learn.fit_one_cycle(epochs, lr)
                elif pretrained:
                    learn.fine_tune(epochs - 5, base_lr=lr, freeze_epochs = 5)
                val_pct.append(learn.validate()[1])
                a,b = learn.get_preds(ds_idx=2)
                tst_preds.append(a)
                run_count+=1
                #del learn
                torch.cuda.empty_cache()
                gc.collect()
            tst_preds_copy = torch.stack(tst_preds.copy())
            run_details = 'arch:' + str(arch) + '--' + 'lr:' + str(lr) + '--' + 'bs:' + str(bs) + '--' + 'image_size:' + str(size) + '--' + 'epochs:' + str(epochs) + '--' + 'optimizer:' + opt + '--' + 'loss_func:' + loss + '--' + 'pretrained:' + str(pretrained) + '--' + 'normalize:' + str(normalize) + '--' + 'augmentation:' + str(aug) 
            with open('/home/jmtzt/Desktop/IC/Results-Report-ic2019/Local-Machine/preds_ddsm_mass/Scenario' + scenario + '/' + str(arch), 'wb') as preds:
                np.save(preds, tst_preds_copy, allow_pickle=True)
             #Get Accuracy for test set - ensemble
            hat = tst_preds[0]
            for pred in tst_preds[1:]:
                hat += pred
            hat /= len(tst_preds)
            print(hat.shape)
            acc_test = accuracy_score(hat.argmax(1), b)
            acc_test_item = acc_test.item()
            #Save accuracy to file with training details!
            print(run_details)
            print("Acc_ensemble: " + str(acc_test.item()))
            with open("results.txt", "a") as resultFile:
                resultFile.write(run_details + '\nAcc_ensemble:' + str(acc_test_item) + '\n')
            #learn.save(run_details + "--stage-1")
            del learn
            torch.cuda.empty_cache()
            gc.collect()