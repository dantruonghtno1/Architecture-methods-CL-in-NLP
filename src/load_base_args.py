from config import set_args

# Arguments
args = set_args()

def load():
    # ============= dataset base ==================
    if args.task == 'asc': #aspect sentiment classication
        args.ntasks = 19
        args.num_train_epochs = 10
        args.xusemeval_num_train_epochs = 10
        args.bingdomains_num_train_epochs = 30
        args.bingdomains_num_train_epochs_multiplier = 3
        args.nepochs = 100
        args.nclasses = 3


    if args.task == 'dsc': #document sentiment classication
        args.ntasks = 10
        args.num_train_epochs = 20
        args.nepochs = 100
        args.nclasses = 2



    # ============= backbone base ==================
    if 'bert_adapter' in args.backbone:
        args.apply_bert_output = True
        args.apply_bert_attention_output = True


    # ============= approach base ==================

    # Some are in the "base" folder


    if args.baseline == 'acl' :
        args.buffer_size = 128
        args.buffer_percent = 0.02
        args.alpha = 0.5
        args.beta = 0.5

    if args.baseline == 'b-cl':
        args.apply_bert_output = True
        args.apply_bert_attention_output = True
        args.build_adapter_capsule_mask = True
        args.apply_one_layer_shared = True
        args.semantic_cap_size = 3

    if args.baseline == 'classic':
        args.apply_bert_output = True
        args.apply_bert_attention_output = True
        args.sup_loss  = True
        args.amix  = True

    if args.baseline == 'ctr':
        args.apply_bert_output = True
        args.apply_bert_attention_output = True
        args.build_adapter_capsule_mask = True
        args.apply_one_layer_shared = True
        args.use_imp = True
        args.transfer_route = True
        args.share_conv = True
        args.larger_as_share = True
        args.adapter_size = True

    # ============= additional for DIL ==================

    if not args.ent_id and not args.last_id and 'dil' in args.scenario:
        args.last_id = True # you have to chose one way to decide the ID


    if args.ent_id:
        args.resume_model = True
        args.eval_only = True
        args.eval_batch_size = 1

    if args.eval_only:
        args.resume_model = True
        if args.ntasks == 10 and not args.eval_each_step:
            args.resume_from_task = 9
        elif args.ntasks == 20 and not args.eval_each_step:
            args.resume_from_task = 19

    if args.unseen:
        args.ntasks_unseen = 10


    #================ analysis base ================

    if args.exit_after_first_task:
        args.nepochs = 1
        args.num_train_epochs = 1
        args.xusemeval_num_train_epochs = 1
        args.bingdomains_num_train_epochs = 1
        args.bingdomains_num_train_epochs_multiplier = 1
        args.eval_only = False

    return args