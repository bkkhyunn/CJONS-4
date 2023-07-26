import os, sys

sys.path.append('/workspace')

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

PATH_DICT = {
    'photo': os.path.join(DATA_DIR, 'photos.json'), 
    'image': os.path.join(DATA_DIR, 'photos'), 
    'item_paths': os.path.join(DATA_DIR, 'yelp_academic_dataset_business.json'), 
    'user_paths': os.path.join(DATA_DIR, 'yelp_academic_dataset_user.json'), 
    'review_paths': os.path.join(DATA_DIR, 'yelp_academic_dataset_review.json') 
}

photo_paths = os.path.join(DATA_DIR, 'photos.json')
img_paths = os.path.join(DATA_DIR, 'photos')
item_paths = os.path.join(DATA_DIR, 'yelp_academic_dataset_business.json')
user_paths = os.path.join(DATA_DIR, 'yelp_academic_dataset_user.json')
review_paths = os.path.join(DATA_DIR, 'yelp_academic_dataset_review.json')