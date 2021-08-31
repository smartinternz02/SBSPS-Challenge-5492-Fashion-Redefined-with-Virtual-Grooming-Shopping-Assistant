from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


app = Flask(__name__)

model = load_model(r'C:\Users\HP\clothes.h5')



@app.route('/')
def index():
	return render_template("index.html", data="hey")


@app.route("/prediction.html", method=["POST"])
def prediction():
    imag = request.files['img']
    img = image.load_img(imag,target_size = (64,64))
    img.save("img.jpg")
    x = image.img_to_array(imag)
    x = np.expand_dims(x,axis = 0)
    pred = model.predict_classes(x)
    index = ['Blazer','Footwear','Hoodies','Men shirts',"Men's bottomwear",'Mens Tshirts','Sweater','girls tops','kurtas','leggings','sarees']
    pred = index[pred[0]]
    return render_template("prediction.html", data=pred)


if __name__ == "__main__":
	app.run(debug=True)