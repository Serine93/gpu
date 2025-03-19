import cv2
import numpy as np
from numba import cuda
import time

image = cv2.imread('images.jpg')

if image is None:
    print("Erreur lors du chargement de l'image!")
    exit()

image = cv2.resize(image, (256, 256))

def simulate_daltonism(image, daltonism_type):
    image_copy = image.copy()

    if daltonism_type == "Deutéranopie":
        image_copy[:, :, 1] = 0  
    elif daltonism_type == "Protanopie":
        image_copy[:, :, 2] = 0 
    elif daltonism_type == "Achromatopsie":
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)

    return image_copy

@cuda.jit
def analyser_pixels_kernel(image, rouge_count, vert_count, bleu_count, daltonien_count):
    x, y = cuda.grid(2)

    if x < image.shape[0] and y < image.shape[1]:
        pixel = image[x, y]
        bleu, vert, rouge = pixel  

        if rouge > vert and rouge > bleu:
            cuda.atomic.add(rouge_count, 0, 1)
        elif vert > rouge and vert > bleu:
            cuda.atomic.add(vert_count, 0, 1)
        elif bleu > rouge and bleu > vert:
            cuda.atomic.add(bleu_count, 0, 1)

        if (abs(int(rouge) - int(vert)) < 50 or abs(int(rouge) - int(bleu)) < 50 or abs(int(vert) - int(bleu)) < 50):
            cuda.atomic.add(daltonien_count, 0, 1)

def main():
    start_time = time.time()

    rouge_count = np.zeros(1, dtype=np.int32)
    vert_count = np.zeros(1, dtype=np.int32)
    bleu_count = np.zeros(1, dtype=np.int32)
    daltonien_count = np.zeros(1, dtype=np.int32)

    image_gpu = cuda.to_device(image)

    threadsperblock = (16, 16)  
    blockspergrid = (int((image.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]),
                     int((image.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]))

    analyser_pixels_kernel[blockspergrid, threadsperblock](image_gpu, rouge_count, vert_count, bleu_count, daltonien_count)

    rouge_count_host = rouge_count[0]
    vert_count_host = vert_count[0]
    bleu_count_host = bleu_count[0]
    daltonien_count_host = daltonien_count[0]

    total_pixels = image.shape[0] * image.shape[1]
    rouge_percentage = (rouge_count_host / total_pixels) * 100
    vert_percentage = (vert_count_host / total_pixels) * 100
    bleu_percentage = (bleu_count_host / total_pixels) * 100
    daltonien_percentage = (daltonien_count_host / total_pixels) * 100

    print(f"Pourcentage de pixels rouges: {rouge_percentage:.2f}%")
    print(f"Pourcentage de pixels verts: {vert_percentage:.2f}%")
    print(f"Pourcentage de pixels bleus: {bleu_percentage:.2f}%")
    print(f"Pourcentage de pixels difficiles à percevoir pour les daltoniens: {daltonien_percentage:.2f}%")

    if daltonien_percentage > 5:
        print("L'image n'est pas bien visible pour les daltoniens.")

        # Exemple de type de daltonisme
        daltonism_type = "Deutéranopie"  
        modified_image = simulate_daltonism(image, daltonism_type)
        cv2.imwrite(f"{daltonism_type}_modified_image.jpg", modified_image)
        print(f"L'image modifiée pour {daltonism_type} a été enregistrée.")
    else:
        print("L'image est bien visible pour les daltoniens.")

    print(f"Temps d'exécution avec CUDA: {time.time() - start_time} secondes")

if __name__ == "__main__":
    main()
