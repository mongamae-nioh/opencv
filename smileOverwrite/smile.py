#参考サイト
#http://blanktar.jp/blog/2015/02/python-opencv-realtime-lauhgingman.html

#coding: utf-8

import cv2
import numpy


cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')  # 顔正面を検出する分類器指定

cam = cv2.VideoCapture(0)  # カメラ起動

laugh = cv2.imread('images/smile.png', -1)  # スマイル画像読み込み
mask = cv2.cvtColor(laugh[:,:,3], cv2.cv.CV_GRAY2BGR)/255.0  # チャンネルだけを抜き出して0から1までの値にする。あと3チャンネルにしておく。
laugh = laugh[:,:,:3]  # αチャンネルはもういらないので消してしまう。

while True:
	ret, img = cam.read()  # カメラから画像を読み込む。

	if not ret:
		print 'error?'
		break

	gray = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)  # 画像認識を高速に行うためにグレースケール化。
	gray = cv2.resize(gray, (img.shape[1]/4, img.shape[0]/4))  # そのままだと遅かったので画像を4分の1にしてさらに高速化。

	faces = cascade.detectMultiScale(gray)  # 顔を探す。

	if len(faces) > 0:
		for rect in faces:
			rect *= 4  # 認識を4分の1のサイズの画像で行ったので、結果は4倍しないといけない。

			#  単純に大きくするとキャプチャした画像のサイズを越えてしまうので少し面倒な処理をしている。
			rect[0] -= min(25, rect[0])
			rect[1] -= min(25, rect[1])
			rect[2] += min(50, img.shape[1]-(rect[0]+rect[2]))
			rect[3] += min(50, img.shape[0]-(rect[1]+rect[3]))

			# 認識した顔と同じサイズにリサイズする。
			laugh2 = cv2.resize(laugh, tuple(rect[2:]))
			mask2 = cv2.resize(mask, tuple(rect[2:]))

			# 合成
			img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = laugh2[:,:] * mask2 + img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] * (1.0 - mask2)

	cv2.imshow('smiling!', img)

	if cv2.waitKey(10) > 0:
		break

cam.release()
cv2.destroyAllWindows()
