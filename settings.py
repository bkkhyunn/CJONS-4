import os, sys

sys.path.append('/workspace')

BASE_DIR = os.path.dirname(__file__)
ORIGIN_DATA_DIR = os.path.join(BASE_DIR, 'dataset')
DATA_DIR = os.path.join(BASE_DIR, 'data')

PATH_DICT = {
    'photo': os.path.join(ORIGIN_DATA_DIR, 'photos.json'), 
    'image': os.path.join(ORIGIN_DATA_DIR, 'photos'), 
    'item_paths': os.path.join(ORIGIN_DATA_DIR, 'yelp_academic_dataset_business.json'), 
    'user_paths': os.path.join(ORIGIN_DATA_DIR, 'yelp_academic_dataset_user.json'), 
    'review_paths': os.path.join(ORIGIN_DATA_DIR, 'yelp_academic_dataset_review.json') 
}

SAVE_DIR = os.path.join(BASE_DIR, 'model_parameters')