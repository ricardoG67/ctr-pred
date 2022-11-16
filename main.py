from flask import Flask, render_template, request, redirect, url_for
import cv2 as cv
from werkzeug.utils import secure_filename
import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

    cuadrantes, imagen_2, kp_sift, kp_fast = extractor(filename)
    
    cv.imwrite(f"static/uploads/2_{filename}", imagen_2)

    cont = 1
    sift = cv.SIFT_create()
    fast = cv.FastFeatureDetector_create()

    for i in cuadrantes:
        gray= cv.cvtColor(i,cv.COLOR_BGR2GRAY)

        kp = sift.detect(gray,None)
        img=cv.drawKeypoints(gray,kp,i)
        cv.imwrite(f"static/uploads/cuadrante{cont}_sift.png", img)

        kp = fast.detect(gray,None)
        img = cv.drawKeypoints(gray, kp, i)
        cv.imwrite(f"static/uploads/cuadrante{cont}_fast.png", img)
 
        cont+=1
    
    c1 = kp_sift[0]
    c2 = kp_sift[1]
    c3 = kp_sift[2]
    c4 = kp_sift[3]
    c5 = kp_sift[4]
    c6 = kp_sift[5]
    c7 = kp_sift[6]
    c8 = kp_sift[7]
    c9 = kp_sift[8]
    c10 =kp_sift[9]
    c11 =kp_sift[10]
    c12 =kp_sift[11]
    c13 =kp_sift[12]
    c14= kp_sift[13]
    c15 =kp_sift[14]
    c16= kp_sift[15]
    c17= kp_sift[16]
    c18= kp_sift[17]
    c19= kp_sift[18]
    c20= kp_sift[19]
    c21= kp_sift[20]
    c22= kp_sift[21]
    c23= kp_sift[22]
    c24= kp_sift[23]
    c25= kp_sift[24]


    c1_fast = kp_fast[0]
    c2_fast = kp_fast[1]
    c3_fast = kp_fast[2]
    c4_fast = kp_fast[3]
    c5_fast = kp_fast[4]
    c6_fast = kp_fast[5]
    c7_fast = kp_fast[6]
    c8_fast = kp_fast[7]
    c9_fast = kp_fast[8]
    c10_fast = kp_fast[9]
    c11_fast = kp_fast[10]
    c12_fast = kp_fast[11]
    c13_fast = kp_fast[12]
    c14_fast= kp_fast[13]
    c15_fast = kp_fast[14]
    c16_fast= kp_fast[15]
    c17_fast= kp_fast[16]
    c18_fast= kp_fast[17]
    c19_fast= kp_fast[18]
    c20_fast= kp_fast[19]
    c21_fast= kp_fast[20]
    c22_fast= kp_fast[21]
    c23_fast= kp_fast[22]
    c24_fast= kp_fast[23]
    c25_fast= kp_fast[24]

    tabla = []
    tabla.append([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,
    c1_fast,c2_fast,c3_fast, c4_fast, c5_fast, c6_fast, c7_fast, c8_fast, c9_fast, c10_fast,c11_fast,c12_fast, 
    c13_fast, c14_fast, c15_fast, c16_fast,c17_fast,c18_fast,c19_fast, c20_fast, c21_fast, c22_fast, c23_fast, c24_fast, c25_fast])
    
    df = pd.DataFrame(tabla, columns=["c1_sift", "c2_sift", "c3_sift", "c4_sift", "c5_sift", "c6_sift", "c7_sift", "c8_sift", "c9_sift",
                                  "c10_sift", "c11_sift", "c12_sift", "c13_sift", "c14_sift", "c15_sift", "c16_sift",
                                  "c17_sift", "c18_sift", "c19_sift", "c20_sift", "c21_sift", "c22_sift", "c23_sift", "c24_sift", "c25_sift",

                                  "c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15","c16","c17","c18","c19","c20",
                                  "c21","c22","c23","c24","c25"])

    x = df.values[:,:]
    escalador = StandardScaler()

    x_norm = escalador.fit_transform(x)
    clf = pickle.load(open("finalized_model.sav", 'rb'))

    predicho = clf.predict(x_norm)

    return render_template('part2.html', filename=filename, predicho=predicho)

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

    keypoints = []
    for i in cuadrantes:
        
        gray= cv.cvtColor(i,cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp = sift.detect(gray,None)

        keypoints.append(len(kp))

    
    kp_fast = []
    for i in cuadrantes:
        
        gray = cv.cvtColor(i,cv.COLOR_BGR2GRAY)
        fast = cv.FastFeatureDetector_create()
        kp = fast.detect(gray,None)

        kp_fast.append(len(kp))

    return cuadrantes, img_original, keypoints, kp_fast

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
    
# @app.template_filter('nanmin')
# def minimo_numpy(value):
#     return np.nanmin(value)

if __name__ == '__main__':
    app.run(debug=True)
