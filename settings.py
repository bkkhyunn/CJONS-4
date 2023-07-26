import os 

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'aclImdb')
YELP_DIR = os.path.join(BASE_DIR, 'dataset')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')



photo_paths = os.path.join(YELP_DIR, 'yelp_photos-2\\photos.json')
item_paths = os.path.join(YELP_DIR, 'yelp_dataset-2\\yelp_academic_dataset_business.json')
user_paths = os.path.join(YELP_DIR, 'yelp_dataset-2\\yelp_academic_dataset_user.json')
review_paths = os.path.join(YELP_DIR, 'yelp_dataset-2\\yelp_academic_dataset_review.json')