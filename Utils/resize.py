import glob
import cv2

# output name
photo_name = 'truck_'

# resize dimensions
dim = (256, 256)

i = 1
for img in glob.glob('C:/Users/Nico/Desktop/Sistemi_operativi/Progetto/Immagini_pre_resize/*.j*g'):
    nextimg = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(nextimg, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite('C:/Users/Nico/Desktop/Sistemi_operativi/Progetto/Sistemi-Operativi-Dedicati/Images/demo/' + photo_name + str(i) + '.jpg', resized)
    i = i + 1

print('Done')
