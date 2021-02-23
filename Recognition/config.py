class Config_eyedoctor(object):

   model = 'resnet18' 
   # here can be changed to T4_224 for faster training.
   train_data_root = './data/eyedoctor/T4/train/'
   val_data_root = './data/eyedoctor/T4/val/'
   test_data_root = './data/eyedoctor/T4/test/'
   inter_1_test_data_root = './data/eyedoctor/T4/s9/' #inter-session test
   inter_2_test_data_root = './data/eyedoctor/T4/s10/' #inter-session test
   ckpt_path = './checkpoints/'

   batch_size = 256 
   num_workers = 4 
     
   max_epoch = 25
   lr = 1e-4 
   # lr_decay = 0.95 
   # weight_decay = 1e-4 

class Config_securitycode_iph6s(object):
   model = 'LeNet_EMAGE' 
   train_data_root = './data/security_code/iphone6s/train/'
   val_data_root = './data/security_code/iphone6s/val/'
   test_data_root = './data/security_code/iphone6s/test/'
   inter_1_test_data_root = './data/security_code/iphone6s/s8/test/' #inter-session test
   inter_2_test_data_root = './data/security_code/iphone6s/s9/test/' #inter-session test
   ckpt_path = './checkpoints/'


   batch_size = 256 
   num_workers = 8 
   print_freq = 20 

   debug_file = '/tmp/debug' 
   result_file = 'result.csv'
   
   max_epoch = 100
   lr = 1e-3 
   lr_decay = 0.1
   decay_steps = 50 


class Config_securitycode_iph6(object):
   model = 'LeNet_EMAGE' 
   train_data_root = './data/security_code/iphone6/train/'
   val_data_root = './data/security_code/iphone6/val/'
   test_data_root = './data/security_code/iphone6/test/'
   ckpt_path = './checkpoints/'



   batch_size = 256 
   num_workers = 8 
   print_freq = 20 

   max_epoch = 100
   lr = 1e-3 
   lr_decay = 0.1
   decay_steps = 70 

class Config_securitycode_honor(object):
   model = 'LeNet_EMAGE' 
   train_data_root = './data/security_code/honor6x/train/'
   val_data_root = './data/security_code/honor6x/val/'
   test_data_root = './data/security_code/honor6x/test/'

   ckpt_path = './checkpoints/'



   batch_size = 256 
   num_workers = 8 
   print_freq = 20 
   
   max_epoch = 100
   lr = 1e-3 
   lr_decay = 0.1
   decay_steps = 70 
