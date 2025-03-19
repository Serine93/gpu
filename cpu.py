import cv2
import numpy as np
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

def analyser_pixels(image):
    rouge_count = 0
    vert_count = 0
    bleu_count = 0
    daltonien_count = 0
    total_pixels = image.shape[0] * image.shape[1]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            bleu, vert, rouge = pixel  

            if rouge > vert and rouge > bleu:
                rouge_count += 1
            elif vert > rouge and vert > bleu:
                vert_count += 1
            elif bleu > rouge and bleu > vert:
                bleu_count += 1

            if (abs(int(rouge) - int(vert)) < 50 or abs(int(rouge) - int(bleu)) < 50 or abs(int(vert) - int(bleu)) < 50):
                daltonien_count += 1

    rouge_percentage = (rouge_count / total_pixels) * 100
    vert_percentage = (vert_count / total_pixels) * 100
    bleu_percentage = (bleu_count / total_pixels) * 100
    daltonien_percentage = (daltonien_count / total_pixels) * 100

    return rouge_percentage, vert_percentage, bleu_percentage, daltonien_percentage

def main():
    start_time = time.time()

    rouge_percentage, vert_percentage, bleu_percentage, daltonien_percentage = analyser_pixels(image)

    print(f"Pourcentage de pixels rouges: {rouge_percentage:.2f}%")
    print(f"Pourcentage de pixels verts: {vert_percentage:.2f}%")
    print(f"Pourcentage de pixels bleus: {bleu_percentage:.2f}%")
    print(f"Pixels difficiles pour les daltoniens: {daltonien_percentage:.2f}%")

    if daltonien_percentage > 5:
        print("L'image n'est pas bien visible pour les daltoniens.")
        time.sleep(0.5) 

        # Exemple de type de daltonisme
        daltonism_type = "Deutéranopie"
        modified_image = simulate_daltonism(image, daltonism_type)
        cv2.imwrite(f"{daltonism_type}_modified_image.jpg", modified_image)
        print(f"L'image modifiée pour {daltonism_type} a été enregistrée.")
    else:
        print("L'image est bien visible pour les daltoniens.")

    print(f"Temps d'exécution sur le CPU: {time.time() - start_time:.6f} secondes")

if __name__ == "__main__":
    main()
