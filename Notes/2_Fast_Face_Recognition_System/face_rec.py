import numpy as np
import pandas as pd
import cv2
import datetime
import redis

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# Connect to Redis Client
hostname = 'redis-12084.c301.ap-south-1-1.ec2.cloud.redislabs.com'
portnumber = 12084
password = 'HnYyQx7B7hqPWS0OvE45nVAMm48xzkRd'

r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)


# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc',root='/Users/shashankpandey/Downloads/Notes/2_Fast_Face_Recognition_System/insightface_model', providers = ['CPUExecutionProvider'])
faceapp.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.5)

# ML Search Algorithm
def ml_search_algorithm(dataframe,feature_column,test_vector,
                        name_role=['Name','Role'],thresh=0.5):
    """
    cosine similarity base search algorithm
    """
    # step-1: take the dataframe (collection of data)
    dataframe = dataframe.copy()
    # step-2: Index face embeding from the dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
    
    # step-3: Cal. cosine similarity
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    # step-4: filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        # step-5: get the person name
        data_filter.reset_index(drop=True,inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
        
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
        
    return person_name, person_role


def face_prediction(test_image, dataframe,feature_column,
                        name_role=['Name','Role'],thresh=0.5):
    # step-1: take the test image and apply to insight face
    results = faceapp.get(test_image)
    test_copy = test_image.copy()
    # step-2: use for loop and extract each embedding and pass to ml_search_algorithm

    for res in results:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res['embedding']
        person_name, person_role = ml_search_algorithm(dataframe,
                                                       feature_column,
                                                       test_vector=embeddings,
                                                       name_role=name_role,
                                                       thresh=thresh)
        if person_name == 'Unknown':
            color =(0,0,255) # bgr
        else:
            color = (0,255,0)


        cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)

        text_gen = person_name
        current_datetime = datetime.datetime.now()
        date = str(current_datetime.date())+str(current_datetime.time())
        # date = str(date)

        person_name = person_name + date

        cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
        cv2.putText(test_copy,date,(x2,y2),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
        # print(date)

    return test_copy

# import datetime

# # Create a datetime object

# # Get the date from the datetime object
# date = current_datetime.date()

# # Print the date
# print(date)
