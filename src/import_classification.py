from config import set_args

# Arguments
args = set_args()

# ----------------------------------------------------------------------
# Language Datasets.
# ----------------------------------------------------------------------

bert_backbone = ['bert','bert_adapter','bert_frozen']
w2v_backbone = ['w2v','w2v_as']

language_dataset = ['asc','dsc','ssc','nli','newsgroup']
image_dataset = ['celeba','femnist','vlcs','cifar10','mnist','fashionmnist','cifar100']


if args.task == 'asc': #aspect sentiment classication
    from dataloaders.asc import bert as dataloader

# elif args.task == 'dsc': #document sentiment classication
#     if args.backbone=='w2v':
#         from dataloaders.dsc import w2v as dataloader
#     elif args.backbone in bert_backbone:  # all others
#         from dataloaders.dsc import bert as dataloader


# ----------------------------------------------------------------------
# Lanaguage approaches.
# ----------------------------------------------------------------------
if args.task in language_dataset:
    
    if args.backbone == 'bert_adapter':
        if args.baseline=='ctr' or  args.import_modulesbaseline=='b-cl':
            from approaches.classification import bert_adapter_capsule_mask as approach
            from networks.classification import bert_adapter_capsule_mask as network
        elif args.baseline=='classic':
            from approaches.classification import bert_adapter_mask as approach
            from networks.classification import bert_adapter_mask as network
