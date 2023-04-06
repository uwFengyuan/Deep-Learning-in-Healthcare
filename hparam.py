class hparams:

    #train_or_test = 'test'
    #output_dir = 'logs/your_program_name'
    loss_functions = ['DiceCE_Loss', 'Dice_Loss', 'CrossE_Loss', 'BCE_Loss', 
                      'Focal_Loss', 'Tversky_Loss', 'DiceFocal_Loss', 'FocalTversky_Loss']
    IOP_probability = [0.6, 0.2, 0.2]
    aug = [False, True] # 是否要augmentation
    pre = [False, True]
    #latest_checkpoint_file = 'checkpoint_latest.pt'
    steps = [3, 4, 5]
    aug_commands = ['Affine', 'BiasField',
                   'Gamma', 'Noise', 'Flip', 'Motion', 'All']
    total_epochs = 50
    #epochs_per_checkpoint = 10
    batch_size = 6
    #ckpt = None # 是否有断点
    init_lr = 0.01
    #scheduer_step_size = 20
    #scheduer_gamma = 0.8
    debug = True # 是否要debug
    #mode = '3d' # '2d or '3d'
    in_channel = 1 # in channel
    out_channel = 1 # output channel
    init_features = 8
    early_stop = 5

    #crop_or_pad_size = 64,128,128 # if 2D: 256,256,1
    #patch_size = 40,32,32 # if 2D: 128,128,1 

    # for test
    #patch_overlap = 8,16,16 # if 2D: 4,4,0

    #old_arch = '*.mhd'

    #save_arch = '.nii.gz'

    files = ['IOP', 'Guys', 'HH']
    image_dir_IOP = 'healthcare_data/train_val_test/X_IOP.npy'
    label_dir_IOP = 'healthcare_data/train_val_test/y_IOP.npy'
    image_dir_Guys = 'healthcare_data/train_val_test/X_Guys.npy'
    label_dir_Guys = 'healthcare_data/train_val_test/y_Guys.npy'
    image_dir_HH = 'healthcare_data/train_val_test/X_HH.npy'
    label_dir_HH = 'healthcare_data/train_val_test/y_HH.npy'
    #source_test_dir = 'healthcare_data/brain_extraction/X_IOP.npy'
    #label_test_dir = 'healthcare_data/brain_extraction/y_IOP.npy'


    #output_dir_test = 'results/your_program_name'