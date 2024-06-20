from flask import Flask, render_template, request, redirect
import numpy as np
import tensorflow
import os




# target_img = os.path.join(os.getcwd() , 'static/images')


class_dict={'Pepper__bell___Bacterial_spot': 0,
 'Pepper__bell___healthy': 1,
 'Potato___Early_blight': 2,
 'Potato___Late_blight': 3,
 'Potato___healthy': 4,
 'Tomato_Bacterial_spot': 5,
 'Tomato_Early_blight': 6,
 'Tomato_Late_blight': 7,
 'Tomato_Leaf_Mold': 8,
 'Tomato_Septoria_leaf_spot': 9,
 'Tomato_Spider_mites_Two_spotted_spider_mite': 10,
 'Tomato__Target_Spot': 11,
 'Tomato__Tomato_YellowLeaf__Curl_Virus': 12,
 'Tomato__Tomato_mosaic_virus': 13,
 'Tomato_healthy': 14}


def prediction_cls(prediction):
    for key, clss in class_dict.items():
        if np.argmax(prediction)==clss:
            return key


app = Flask(__name__, template_folder='templates')
model = tensorflow.keras.models.load_model("/archive/vgg16_finetune_model.h5")


@app.route('/')
def create_view():
    return render_template('plant.html')

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        imageFile = request.files['imageFile']
       
       
        image_path = '/archive/test/' + imageFile.filename
    


        image = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tensorflow.keras.preprocessing.image.img_to_array(image)
        image= np.expand_dims(image, axis=0)
        image = image/255
        
        classification = prediction_cls(model.predict(image))
        
        return render_template('predict.html', predictclass = classification)
    
    else:
        return "unable to read the file"


if __name__ == '__main__':
   app.run(port=5000,use_reloader=False, debug=True)





















 # if imageFile and allowed_file(imageFile.filename):
        #     filename = imageFile.filename
        #     file_path = os.path.join('static/images', filename)
        #     imageFile.save(file_path)

