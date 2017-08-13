# directory
main_dir    = '/path/to/main/directory' # you need to change this
aug_dir     = main_dir + '/Augmented/'
cnn_dir     = main_dir + '/CNN/'
code_dir    = main_dir + '/Code/'
images_dir  = main_dir + '/Images/'
papers_dir  = main_dir + '/Papers/'
results_dir = main_dir + '/Results/'
test_dir    = main_dir + '/Test/'

# number of surrogate classes
num_surr    = 100

# number of augmentations per image
num_aug     = 200

# patches
patch_size  = 96
height      = 64
width       = 64

# CNN
cnn_size    = '/Small'      # '/Big'
cnn_current = cnn_dir + cnn_size + '/' + str(num_surr)
cnn_models  = cnn_current + '/models'
cnn_model   = cnn_models + '/model'

# train, validation and labels
train_dir       = cnn_current + '/train/'
validate_dir    = cnn_current + '/validate/'
labels_file     = cnn_current + '/mylabels.txt'

# features and results
results_current = results_dir + cnn_size + '/' + str(num_surr)
features_file   = results_current + '/features.txt'
results_file    = results_current + '/results.csv'

# training
train_steps     = 5000
learning_rate   = 0.0001
batch_size_     = 250       # 500

# k-means
num_clusters    = 20

# visualization
visual_file     = train_dir + '/label_74' 
