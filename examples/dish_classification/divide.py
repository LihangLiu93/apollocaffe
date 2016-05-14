import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

folder = './dishes_dataset_316'

classes = sorted(os.listdir(folder))

with open('classes.lst','w') as f:
	f.write('\n'.join(classes))
	print len(classes),'Classes'#,classes[0]

ftest = open('imglist_test_316.txt','w')
ftrain = open('imglist_train_316.txt','w')
count  = 0
flag = True
for idx, classname in enumerate(classes):
	print classname
	class_path = os.path.join(folder,classname)
	for i, imgname in enumerate(os.listdir(class_path)):
		imgdir = os.path.join(class_path,imgname)
		count += 1
		try:
			img = plt.imread(imgdir)
			plt.imshow(img)
			#plt.close()
		except Exception as e:
			print count, imgdir
			flag = False
		if flag:
			if i%10==0:
				ftest.write('%s %d\n'%(imgdir, idx))
			else:
				ftrain.write('%s %d\n'%(imgdir, idx))
			#outfile.flush()
		flag = True
		plt.close()
print count
ftest.close()
ftrain.close()
