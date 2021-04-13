import numpy as np
import cv2
import os
import random
import tqdm
import time



BASE_DIR = "C:/Users/aripa/PycharmProjects/bg_removal"
os.chdir(BASE_DIR)


def remove_bg(path):
    # Parameters
    blur = 21  # "smoothness" of the fg-bg dividing line
    canny_low = 15  # min intesity along which edges are drawn, too low -> too many edges
    canny_high = 150  # max intesity along which edges are drawn, contrast above this value -> edge
    min_area = 0.0005  # min area that a contour in fg can occupy [0, 1]
    max_area = 0.95  # max area that a contour in fg can occupy [0, 1]

    img = cv2.imread(path)

    # Convert image to grayscale
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Dection
    edges = cv2.Canny(image_gray, canny_low, canny_high)

    # optional
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # Get the area of the image as a comparison
    image_area = img.shape[0] * img.shape[1]

    # get the contours and their areas
    contour_info = [(c, cv2.contourArea(c)) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]

    # calculate max and min areas in terms of pixels
    max_area = max_area * image_area
    min_area = min_area * image_area

    # Set up mask with a matrix of 0's
    mask = np.ones(img.shape, dtype=np.uint8) * 255

    # Go through and find relevant contours and apply to mask
    for contour in contour_info:
        # Instead of worrying about all the smaller contours,
        # if the area is smaller than the min, the loop will break
        if min_area < contour[1] < max_area:
            # Add contour to mask
            mask = cv2.fillConvexPoly(mask, contour[0], (0, 0, 0))

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # negate mask
    mask = 255 - mask

    # apply morphology to remove isolated extraneous noise
    # use border constant of black since foreground touches the edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)

    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask

    return result


def overlay_png(foreground, background, pos=(0,0)):
    # function:: alpha blending: overlay a given png image (RGBA) over given background
    # foreground:: foreground image
    # background:: background image
    # position:: position where the png should be attached
    # output:: image result

    h,w,_ = np.shape(foreground)
    rows,cols,_ = np.shape(background)
    y,x = pos[0],pos[1]

    # apply alpha blending pixel by pixel
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(foreground[i][j][3] / 255.0)  # take pixel's alpha and make it in rage (0, 1)
            background[x+i][y+j] = alpha*(foreground[i][j][:3])+(1-alpha)*(background[x+i][y+j])     # alpha blending

    return background


def random_img(folder):
    images = os.listdir(folder)
    max = len(images)
    rnd = random.randrange(0, max)
    path = folder + '/' + images[rnd]
    img = cv2.imread(path)
    return img


def bg_processing(images_folder, dest_folder, bg_folder=None, overlay=True, png_folder=None, save_png=True):
    """
    Prende una cartella composta da immagini (images_folder) crea i png di quelle immagini
    (se save_png == True li salva nella cartella png_folder), e se overlay == True fa l'overlay con
    una img casuale di background presa da bg_folder
    :param images_folder: cartella sorgente, solo immagini
    :param dest_folder: cartella destinazione, per png e jpng
    :param bg_folder: cartella dei backgrounds
    :param overlay: Se True effettua l'overlay
    :param png_folder: cartella destinazione per i png
    :param save_png: Se True salva i png in png_folder
    :return: 0
    """

    # Controllo sui parametri

    if overlay:
        if bg_folder is None:
            print('Per creare immagini con nuovo background è necessario specificare il campo "bg_folder".')
            return 1

    if save_png:
        if png_folder is None:
            print('Per salvare i png è necessario specificare il campo "png_folder".')
            return 1

    # Prende in una lista tutti i nomi dei file nella cartella images_folder
    images = os.listdir(images_folder)

    # Crea una lista vuota dove andrò a tenere i png delle immagini senza background
    pngs = []

    # Crea i png (nparray) e li salva nella lista pngs
    print("Inizio rimozione background immagini in " + images_folder + '...')
    time.sleep(1)
    # Display della progress bar
    LENGTH = len(images)  # Numero di iterazioni per fillare pbar
    pbar = tqdm.tqdm(total=LENGTH)  # Init pbar

    for path in images:
        fname, _ = path.split('.')
        path = images_folder + '/' + fname + '.jpg'
        png = remove_bg(path)

        # se save_png==True:: salva nella cartella designata
        if save_png:
            path = png_folder + '/' + fname + '.png'
            cv2.imwrite(path, png)

        # il risultato è mantenuto in una coppia con il nome del file (senza estensione)
        # sottoforma di stringa e l'immagine sottoforma di nparray (RGBA)
        result = (fname, png)
        pngs.append(result)

        pbar.update(n=1)    # Incrementa contatore pbar
        pbar.refresh()

    pbar.close()

    # Se overlay == True, effettua overlay dei png contenuti nella lista pngs
    # con un background scelto casualmente nella cartella bg_folder
    if overlay:
        # Display della progress bar
        print("Inizio overlay...")
        time.sleep(1)

        LENGTH = len(pngs)
        pbar = tqdm.tqdm(total=LENGTH)

        for img in pngs:
            fname = img[0]
            foreground = img[1]
            background = random_img(bg_folder)

            # Controlla se bg e fg hanno dimensioni diverse
            # Se sì, ridimensiona la size del bg per eguagliare il fg
            if np.shape(foreground)[:2] != np.shape(background)[:2]:
                h, w = np.shape(foreground)[:2]
                background = cv2.resize(background, (w, h))

            # Esegue overlay e salva il risultato nella directory specificata
            result = overlay_png(foreground, background)
            path = dest_folder + '/' + fname + '.jpg'
            cv2.imwrite(path, result)

            pbar.update(n=1)
            pbar.refresh()

        pbar.close()
        print("Terminato.")
        time.sleep(1)


def main():
    directories = os.listdir('ds_trashnet_resized')
    for dir in directories:
        dest_folder = 'ds_trashnet_new_bg_compost/' + dir
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        png_folder = 'ds_trashnet_png_compost/' + dir
        if not os.path.exists(png_folder):
            os.makedirs(png_folder)

        dir = 'ds_trashnet_resized/' + dir
        bg_processing(dir, dest_folder=dest_folder, bg_folder='backgrounds', png_folder=png_folder)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
