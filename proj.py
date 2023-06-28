import cv2
import numpy as np
import random
import string
from tkinter import *
from tkinter import filedialog
import os
from PIL import Image
from scipy.signal import fftconvolve


dna = {}
dnanot={}
dna["00"] = "A"
dna["01"] = "T"
dna["10"] = "G"
dna["11"] = "C"
dna["A"] = [0, 0]
dna["T"] = [0, 1]
dna["G"] = [1, 0]
dna["C"] = [1, 1]
# DNA xor
dna["AA"] = dna["TT"] = dna["GG"] = dna["CC"] = "A"
dna["AG"] = dna["GA"] = dna["TC"] = dna["CT"] = "G"
dna["AC"] = dna["CA"] = dna["GT"] = dna["TG"] = "C"
dna["AT"] = dna["TA"] = dna["CG"] = dna["GC"] = "T"



def factor(m):
    key1=[]
    for i in range(2,m+1):
        if(m%i==0):
            key1.append(i)
    return key1

def keygen(x,r,size):
    key=[]
    index=[]
    for i in range(size):
        x=r*x*(1-x)
        key.append(x)
        index.append(i)
    for i in range(size):
        for j in range(size):
            if(key[i]>key[j]):
                key[i],key[j]=key[j],key[i]
                index[i],index[j]=index[j],index[i]
    return index

def shuffle(m,n,b,g,r,t,x1,y1,w1,h1):
    m, n = b.shape
    w1=w1+10-w1%10
    h1=h1+10-h1%10
    key1= factor(w1)
    #print(key1)
    length=len(key1)
    x=int(length/10*t)
    x=key1[x]
    B1 = b
    G1 = g
    R1 = r
    B2=b
    G2=g
    R2=r
    #print(B1.shape, b.shape, x)
    key = []
    key = keygen(0.01, 3.95, x)
    #print(m,x)
    print("x1==",x1,y1,h1,w1)
    for i in range(x1, x1+w1, x):
        for k in range(0, x):
            for j in range(y1,y1+h1):
                B1[i+k,j] = b[i + key[k],j]
                G1[i+k,j] = g[i + key[k],j]
                R1[i+k,j] = r[i + key[k],j]
    key1 = factor(h1)
    #print(key1)
    length = len(key1)
    y = int(length / 10 * t)
    y = key1[y]
    key = keygen(0.01, 3.95, y)
    for i in range(x1,x1+w1):
        for j in range(y1,y1+h1,y):
            for k in range(y):
                B2[i,j+k]=B1[i,j+key[k]]
                G2[i, j + k] = G1[i, j + key[k]]
                R2[i, j + k] = R1[i, j + key[k]]

    return B2,G2,R2


def dna_encode(b, g, r):
    b = np.unpackbits(b, axis=1)
    g = np.unpackbits(g, axis=1)
    r = np.unpackbits(r, axis=1)
    m, n = b.shape
    r_enc = np.chararray((m, int(n / 2)))
    g_enc = np.chararray((m, int(n / 2)))
    b_enc = np.chararray((m, int(n / 2)))

    for color, enc in zip((b, g, r), (b_enc, g_enc, r_enc)):
        idx = 0
        for j in range(0, m):
            for i in range(0, n, 2):
                enc[j, idx] = dna["{0}{1}".format(color[j, i], color[j, i + 1])]
                idx += 1
                if (i == n - 2):
                    idx = 0
                    break

    b_enc = b_enc.astype(str)
    g_enc = g_enc.astype(str)
    r_enc = r_enc.astype(str)
    return b_enc, g_enc, r_enc

def xor_operationorig(b, g, r, mk):
    m, n = b.shape
    bx = np.chararray((m, n))
    gx = np.chararray((m, n))
    rx = np.chararray((m, n))
    b = b.astype(str)
    g = g.astype(str)
    r = r.astype(str)
    for i in range(0, m):
        for j in range(0, n):
            bx[i, j] = dna["{0}{1}".format(b[i, j], mk[i, j])]
            gx[i, j] = dna["{0}{1}".format(g[i, j], mk[i, j])]
            rx[i, j] = dna["{0}{1}".format(r[i, j], mk[i, j])]

    bx = bx.astype(str)
    gx = gx.astype(str)
    rx = rx.astype(str)

    return bx, gx, rx

def xor_operationchanged(b,g,r,mk,x1,y1,w1,h1):
    m, n = b.shape
    bx = np.chararray((m, n))
    gx = np.chararray((m, n))
    rx = np.chararray((m, n))
    b = b.astype(str)
    g = g.astype(str)
    r = r.astype(str)
    list1=[0,1,2]
    for i in range(0,m):
        for j in range(0,n):
            bx[i,j]=str(b[i,j])
            gx[i,j]=str(g[i,j])
            rx[i,j]=str(r[i,j])
    print(bx.shape)
    print("xfinal=",x1,y1,w1,h1)
    for i in range(x1, x1+w1,2):
        for j in range(3*y1, 3*y1+4*h1,3):
            randval=random.choice(list1)
            if randval==0:
                bx[i, j] = dna["{0}{1}".format(b[i, j], mk[i, j])]
            elif randval==1:
                gx[i, j] = dna["{0}{1}".format(g[i, j], mk[i, j])]
            elif randval==2:
                rx[i, j] = dna["{0}{1}".format(r[i, j], mk[i, j])]
    gx = gx.astype(str)
    bx = bx.astype(str)

    rx = rx.astype(str)
    return bx, gx, rx

def keymatrixencode(key, b):
    b = np.unpackbits(b, axis=1)
    m, n = b.shape
    key_bin = bin(int(key, 16))[2:].zfill(256)
    Mk = np.zeros((m, n), dtype=np.uint8)
    x = 0
    for j in range(0, m):
        for i in range(0, n):
            Mk[j, i] = key_bin[x % 256]
            x += 1

    Mk_enc = np.chararray((m, int(n / 2)))
    idx = 0
    for j in range(0, m):
        for i in range(0, n, 2):
            if idx == (n / 2):
                idx = 0
            Mk_enc[j, idx] = dna["{0}{1}".format(Mk[j, i], Mk[j, i + 1])]
            idx += 1
    Mk_enc = Mk_enc.astype(str)
    return Mk_enc

def dna_decode(b, g, r):
    m, n = b.shape
    r_dec = np.ndarray((m, int(n * 2)), dtype=np.uint8)
    g_dec = np.ndarray((m, int(n * 2)), dtype=np.uint8)
    b_dec = np.ndarray((m, int(n * 2)), dtype=np.uint8)
    for color, dec in zip((b, g, r), (b_dec, g_dec, r_dec)):
        for j in range(0, m):
            for i in range(0, n):
                dec[j, 2 * i] = dna["{0}".format(color[j, i])][0]
                dec[j, 2 * i + 1] = dna["{0}".format(color[j, i])][1]
    b_dec = (np.packbits(b_dec, axis=-1))
    g_dec = (np.packbits(g_dec, axis=-1))
    r_dec = (np.packbits(r_dec, axis=-1))
    return b_dec, g_dec, r_dec

def decompose_matrix(image):
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]
    for values, channel in zip((red, green, blue), (2, 1, 0)):
        img = np.zeros((values.shape[0], values.shape[1]), dtype=np.uint8)
        img[:, :] = (values)
        if channel == 0:
            B = np.asmatrix(img)
        elif channel == 1:
            G = np.asmatrix(img)
        else:
            R = np.asmatrix(img)
    return B, G, R

def recover_image(b, g, r, iname):
    img = cv2.imread(iname)
    iname=iname[:-4]
    print(iname)
    img[:, :, 2] = r
    img[:, :, 1] = g
    img[:, :, 0] = b
    return img


def face_detect(image):
    path=os.getcwd()+"\\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4
    )
    if len(faces)==0:
        print("no faces found")
        exit()
    else:
        return faces[0]


def wiener_adaptive_filter(image, noise, window_size):
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = image[max(0, i - window_size // 2):min(image.shape[0], i + window_size // 2 + 1),
                     max(0, j - window_size // 2):min(image.shape[1], j + window_size // 2 + 1)]
            win_shape = window.shape
            H = np.zeros(win_shape)
            for k in range(win_shape[0]):
                for l in range(win_shape[1]):
                    x = np.array([window[k, l]])
                    X = np.fft.fft2(x, s=noise.shape)
                    G = np.conj(noise) / (np.abs(noise) ** 2 + np.max(noise) * 1e-3)  # Wiener filter
                    H[k, l] = np.real(np.fft.ifft2(G * X))
            filtered_window = fftconvolve(window, H, mode='same')
            filtered_image[i, j] = filtered_window[win_shape[0] // 2, win_shape[1] // 2]

    return filtered_image


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    path=os.getcwd()
    path = filedialog.askopenfilename()
    print("path=", path)
    image = cv2.imread(path)

    x,y,w,h=face_detect(image)
    shuffleval = 5

    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]
    for values, channel in zip((red, green, blue), (2, 1, 0)):
        img = np.zeros((values.shape[0], values.shape[1]), dtype=np.uint8)
        img[:, :] = (values)
        if channel == 0:
            B = np.asmatrix(img)
        elif channel == 1:
            G = np.asmatrix(img)
        else:
            R = np.asmatrix(img)

    img = Image.open(path)
    m, n = img.size
    print("m=",m,n)

    b, g, r = shuffle(m, n, B, G, R, shuffleval, x, y, w, h)
    img=recover_image(b, g, r, path)
    """cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    blurmat=img[x:x+w,y:y+h]
    blurmat=cv2.blur(blurmat,(3,3))
    img[x:x+w,y:y+h]=blurmat
    """cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(("enc1.jpg"), img)
    im = Image.open("enc1.jpg")
    im.show()"""

    random_ints = [random.randint(0, 255) for _ in range(50)]
    pass1 = ''.join([chr(i) for i in random_ints])
    #pass1=''.join(random.choices(string.ascii_letters+string.digits+string.punctuation, k=10))
    key=""
    for i in pass1:
        key = key + str(ord(i))
    key = hex(int(key))

    blue, green, red = decompose_matrix(img)
    Mk_e = keymatrixencode(key, blue)
    blue_e, green_e, red_e = dna_encode(blue, green, red)

    blue_final1, green_final1, red_final1 = xor_operationorig(blue_e, green_e, red_e, Mk_e)
    blue_final2, green_final2, red_final2 = xor_operationchanged(blue_e, green_e, red_e, Mk_e, x, y, w, h)
    b1, g1, r1 = dna_decode(blue_final1, green_final1, red_final1)
    b2, g2, r2 = dna_decode(blue_final2, green_final2, red_final2)
    img1=recover_image(b1,g1,r1,path)
    img2=recover_image(b2,g2,r2,path)
    """cv2.imshow("image", img1)
    cv2.waitKey(0)"""
    cv2.imshow("image", img2)
    cv2.waitKey(0)
    cv2.imwrite("DNAencoded.jpg",img2)
    adpweinerimg=img2[x:x+w,y:y+h]
    kernel=np.array([5,5])
    adpweinerimg=  wiener_adaptive_filter(adpweinerimg, -1, kernel)
    img2[x:x + w, y:y + h]=adpweinerimg
    cv2.imshow("image", img2)
    cv2.waitKey(0)
    cv2.imwrite("final.jpg", img2)



