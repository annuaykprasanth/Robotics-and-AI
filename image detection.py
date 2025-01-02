from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def load_and_detect(image_path):
    
    model = YOLO('yolov8n.pt') 
    
   
    results = model(image_path)

   
    for result in results:
       
        detected_objects = [model.names[int(class_id)] for class_id in result.boxes.cls]
        print(f"Detected objects: {detected_objects}")

   
    annotated_image = results[0].plot()
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Detected Objects")
    plt.show()

if __name__ == "__main__":
    image_path = r"C:\Users\HP\Downloads\simba.jpg" 
    load_and_detect(image_path)