from PIL import Image
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plot
import pytesseract

'''Method to determine the license plate'''
def detectplate(image,blur,v):
    # Receives image from function
    img = cv2.imread(image)
    img2 = img

    #Resizes image for standardization
    resized = imutils.resize(img , width=600, height=600)
    resized2 = imutils.resize(img2, width=600,height=600)

    #Converts resized image into a black and white image
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Filters image to have less noise, makes image smoother for easier detection of contours
    d, sigmaColor, sigmaSpace = 10, 15, 15
    filtered = cv2.bilateralFilter(grayscale, d, sigmaColor, sigmaSpace)

    #get image height and width
    height, width, channels = resized.shape
    print(width,height)

    #Blurs right quadrant of picture
    if (blur):
        topLeft = (0, 0)
        bottomRight = (int(width/v), height)
        x, y = topLeft[0], topLeft[1]
        w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
        ROI = filtered[y:y + h, x:x + w]
        blur = cv2.GaussianBlur(ROI, (51, 51), 0)
        # Insert ROI back into image
        filtered[y:y + h, x:x + w] = blur

        #Blurs left quadrant of picture
        topLeft = (0,(v-1)*int(width/v))
        bottomRight = (int(height), int(width))
        x, y = topLeft[0], topLeft[1]
        w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
        ROI = filtered[x:x + w,y:y + h]
        blur = cv2.GaussianBlur(ROI, (51, 51), 0)
        # Insert ROI back into image
        filtered[x:x + w,y:y + h] = blur

    #Contours objects in grayscaled and filtered image
    lower = 170
    upper = 200
    contour = cv2.Canny(filtered,lower,upper)

    #Displays each step of process
    #cv2.imshow("Grayscale",grayscale)
    cv2.imshow("Filtered",filtered)
    cv2.imshow("Contoured",contour)

    #Get contours from contoured image
    contours, hierarchy = cv2.findContours(contour.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    #If image does not seem to contain license plate prints message and returns
    detect = False
    if (len(contours)==0):
        print("ALERT: Image does not contain a detectable contours")
        return
    #Otherwise, program will try to determine the contour for the license plate
    else:
        highest = 0
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01*perimeter, True)
            #Displays contour dots on original image
            approx_dots = cv2.drawContours(resized2, approx, -1, (0, 0, 255), 3)
            cv2.imshow("Approximation",approx_dots)
            if (detectshape(approx)=="rectangle") or (detectshape(approx) == "square"):
            #If 4 edges, assumed as license plate
                target = len(approx)
                if target== 4:
                    platenumber = approx
                    detect = True
                    break

                elif target>4:
                    if highest==0:
                        highest=target
                        otherplate = approx
                    elif target<highest:
                        highest = target
                        otherplate = approx

                else:
                    print("Couldn't detect license plate")
                    return

        if (detect): # if contour count is 4 exactly
            cv2.drawContours(resized, [platenumber], -1, (57, 255, 20), 2)
            print(detectshape(platenumber))
            cv2.imshow("Output", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return platenumber,resized

        else: # If 4 was not present, look for highest count contour
            cv2.drawContours(resized, [otherplate], -1, (57, 255, 20), 2)
            print(detectshape(otherplate))
            cv2.imshow("Output", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return otherplate,resized

def cropandmask(image,plate):
    #Creating mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array(plate, dtype=np.int32)
    #Filling image
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    cv2.imshow("Cropped",masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return masked_image

def detectshape(c):
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif len(approx) == 5:
        shape = "pentagon"
    else:
        shape = "circle"
    return shape

def readplate(image):
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
    newimage = np.asarray(image)
    plate = pytesseract.image_to_string(newimage)
    return plate

def main():
    plt,img = detectplate('detected/image19.jpg',True,4)
    masked = cropandmask(img,plt)
    print(readplate(masked))

if __name__ == '__main__':
   main()



