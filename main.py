from flask import Flask, render_template, request, redirect, url_for
import cv2 as cv
from werkzeug.utils import secure_filename
import os
import numpy as np
#https://stackoverflow.com/questions/47515243/reading-image-file-file-storage-object-using-opencv
#https://tutorial101.blogspot.com/2021/04/python-flask-upload-and-display-image.html

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ctr_prediction', methods=['POST'])
def ctr_prediction():
    imagen = request.files['imagen']
    filename = secure_filename(imagen.filename)
    imagen.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    cuadrantes, imagen_2 = extractor(filename)
    
    cv.imwrite(f"static/uploads/2_{filename}", imagen_2)
    cont = 1
    for i in cuadrantes:
        cv.imwrite(f"static/uploads/cuadrante{cont}.png", i)
        cont+=1

    return render_template('part2.html')

def extractor(imagen):
    img_original = cv.imread(f"static/uploads/{imagen}")

    vertical = (img_original.shape)[0]
    vertical = int(vertical)

    horizontal = (img_original.shape)[1]
    horizontal = int(horizontal)

    # Puntos
    vertical_p1 = vertical/5
    vertical_p2 = (2*vertical)/5
    vertical_p3 = (3*vertical)/5
    vertical_p4 = (4*vertical)/5

    horizontal_p1 = (horizontal)/5
    horizontal_p2 = (2*horizontal)/5
    horizontal_p3 = (3*horizontal)/5
    horizontal_p4 = (4*horizontal)/5

    # Cuadrante1:
    cuadrante_1 = img_original[0:int(vertical_p1), 0:int(horizontal_p1)]

    # Cuadrante2:
    cuadrante_2 = img_original[0:int(vertical_p1), int(
        horizontal_p1):int(horizontal_p2)]

    # Cuadrante3:
    cuadrante_3 = img_original[0:int(vertical_p1), int(
        horizontal_p2):int(horizontal_p3)]

    # Cuadrante4:
    cuadrante_4 = img_original[0:int(vertical_p1), int(
        horizontal_p3):int(horizontal_p4)]

    # Cuadrante5:
    cuadrante_5 = img_original[0:int(vertical_p1), int(
        horizontal_p4):int(horizontal)]

    #########

    # Cuadrante6:
    cuadrante_6 = img_original[int(vertical_p1):int(
        vertical_p2), 0:int(horizontal_p1)]

    # Cuadrante7:
    cuadrante_7 = img_original[int(vertical_p1):int(
        vertical_p2), int(horizontal_p1):int(horizontal_p2)]

    # Cuadrante8:
    cuadrante_8 = img_original[int(vertical_p1):int(
        vertical_p2), int(horizontal_p2):int(horizontal_p3)]

    # Cuadrante9:
    cuadrante_9 = img_original[int(vertical_p1):int(
        vertical_p2), int(horizontal_p3):int(horizontal_p4)]

    # Cuadrante10:
    cuadrante_10 = img_original[int(vertical_p1):int(
        vertical_p2), int(horizontal_p4):int(horizontal)]

    #########

    # Cuadrante11:
    cuadrante_11 = img_original[int(vertical_p2):int(
        vertical_p3), 0:int(horizontal_p1)]

    # Cuadrante12:
    cuadrante_12 = img_original[int(vertical_p2):int(
        vertical_p3), int(horizontal_p1):int(horizontal_p2)]

    # Cuadrante13:
    cuadrante_13 = img_original[int(vertical_p2):int(
        vertical_p3), int(horizontal_p2):int(horizontal_p3)]

    # Cuadrante14:
    cuadrante_14 = img_original[int(vertical_p2):int(
        vertical_p3), int(horizontal_p3):int(horizontal_p4)]

    # Cuadrante15:
    cuadrante_15 = img_original[int(vertical_p2):int(
        vertical_p3), int(horizontal_p4):int(horizontal)]

    #########

    # Cuadrante16:
    cuadrante_16 = img_original[int(vertical_p3):int(
        vertical_p4), 0:int(horizontal_p1)]

    # Cuadrante17:
    cuadrante_17 = img_original[int(vertical_p3):int(
        vertical_p4), int(horizontal_p1):int(horizontal_p2)]

    # Cuadrante18:
    cuadrante_18 = img_original[int(vertical_p3):int(
        vertical_p4), int(horizontal_p2):int(horizontal_p3)]

    # Cuadrante19:
    cuadrante_19 = img_original[int(vertical_p3):int(
        vertical_p4), int(horizontal_p3):int(horizontal_p4)]

    # Cuadrante20:
    cuadrante_20 = img_original[int(vertical_p3):int(
        vertical_p4), int(horizontal_p4):int(horizontal)]

    #########

    # Cuadrante21:
    cuadrante_21 = img_original[int(vertical_p4):(
        vertical), 0:int(horizontal_p1)]

    # Cuadrante22:
    cuadrante_22 = img_original[int(vertical_p4):(
        vertical), int(horizontal_p1):int(horizontal_p2)]

    # Cuadrante23:
    cuadrante_23 = img_original[int(vertical_p4):(
        vertical), int(horizontal_p2):int(horizontal_p3)]

    # Cuadrante24:
    cuadrante_24 = img_original[int(vertical_p4):(
        vertical), int(horizontal_p3):int(horizontal_p4)]

    # Cuadrante25:
    cuadrante_25 = img_original[int(vertical_p4):(
        vertical), int(horizontal_p4):int(horizontal)]

    cuadrantes = [cuadrante_1, cuadrante_2, cuadrante_3, cuadrante_4, cuadrante_5,
                  cuadrante_6, cuadrante_7, cuadrante_8, cuadrante_9, cuadrante_10,
                  cuadrante_11, cuadrante_12, cuadrante_13, cuadrante_14, cuadrante_15,
                  cuadrante_16, cuadrante_17, cuadrante_18, cuadrante_19, cuadrante_20,
                  cuadrante_21, cuadrante_22, cuadrante_23, cuadrante_24, cuadrante_25]

    cv.line(img_original, (0,int(horizontal_p1)), (horizontal, int(horizontal_p1)), (0,0,255), 2)
    cv.line(img_original, (0,int(horizontal_p2)), (horizontal, int(horizontal_p2)), (0,0,255), 2)
    cv.line(img_original, (0,int(horizontal_p3)), (horizontal, int(horizontal_p3)), (0,0,255), 2)
    cv.line(img_original, (0,int(horizontal_p4)), (horizontal, int(horizontal_p4)), (0,0,255), 2)

    cv.line(img_original, (int(vertical_p1),0), (int(vertical_p1), vertical), (0,0,255), 2)
    cv.line(img_original, (int(vertical_p2),0), (int(vertical_p2), vertical), (0,0,255), 2)
    cv.line(img_original, (int(vertical_p3),0), (int(vertical_p3), vertical), (0,0,255), 2)
    cv.line(img_original, (int(vertical_p4),0), (int(vertical_p4), vertical), (0,0,255), 2)

    return cuadrantes, img_original

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
    
# @app.template_filter('nanmin')
# def minimo_numpy(value):
#     return np.nanmin(value)

if __name__ == '__main__':
    app.run(debug=True)
