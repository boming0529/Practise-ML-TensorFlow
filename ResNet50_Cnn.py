from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.applications.resnet50 import preprocess_input, decode_predictions
from tool.read_file import  *
import os

model = ResNet50()

# for testing
# image_paths = ['image/001.jpg', 'image/002.jpg']
# images = [read_image(path) for path in image_paths]

images_dir = 'image'
image_paths = [os.path.join(images_dir, fname) for fname in os.listdir(images_dir) if fname.endswith('.jpg')]


is_show_image = False

if is_show_image:
    for img in images:
        plt.imshow(img)
        plt.axis('off')
        plt.show()

# img_input = np.expand_dims(images, axis=0)  # add batch dimension
img_batch = np.stack(images, axis=0)  # stack images to create a batch
inp = preprocess_input(img_batch)

y_pred = model.predict(inp)

for idx, pred in enumerate(y_pred):
    print(f'\nImage {image_paths[idx]} Prediction:')
    y_pred_type = np.argmax(pred, axis=-1)  # get max index
    print(f'ResNet Recognition Type: {y_pred_type}')
    
    # decode（top-1）
    decoded_preds = decode_predictions(np.expand_dims(pred, axis=0), top=1)[0]
    for i, decoded_pred in enumerate(decoded_preds, 1):
        print(f'{i}. {decoded_pred[1]} ({decoded_pred[0]})')