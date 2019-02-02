from utils import *
from config import *
import argparse as parser

from keras.models import load_model
from keras.applications.xception import preprocess_input
from moviepy.editor import VideoFileClip
from load_data import *

def pipeline_final(img, is_video):
    channel = 1 if is_video else 4
    size = img.shape
    img = cv2.resize(img, dsize=(608, 608))
    img = np.array([img])
    t = model.predict([img, img])
    output_image = reverse_one_hot(t)
    out_vis_image = colour_code_segmentation(output_image, label_values)
    a = cv2.cvtColor(np.uint8(out_vis_image[0]), channel)
    b = cv2.cvtColor(np.uint8(img[0]), channel)
    added_image = cv2.addWeighted(a, 1, b, 1, channel)
    added_image = cv2.resize(added_image, dsize=(size[1],size[0]))

    return added_image

def pipeline_video(img):
    return pipeline_final(img, True)

def pipeline_img(img):
    return pipeline_final(img, False)

def process(media_dir, save_dir, model_dir):
    global model, label_values

    model = load_model(model_dir, custom_objects={'preprocess_input': preprocess_input})
    label_values, _, _ = get_label_values()

    try:
        img = load_image(media_dir)
        output = os.path.join(save_dir, 'output_image.png')
        img = pipeline_img(img)
        cv2.imwrite(output, img)
    except Exception as ex:
        output = os.path.join(save_dir, 'output_video.mp4')
        clip1 = VideoFileClip(media_dir)
        white_clip = clip1.fl_image(pipeline_video)
        white_clip.write_videofile(output, audio=False)

if __name__ == '__main__':

    if __name__ == "__main__":
        args = parser.ArgumentParser(description='Model prediction arguments')

        args.add_argument('-media', '--media_dir', type=str,
                          help='Media Directorium for prediction (mp4,png)')

        args.add_argument('-save', '--save_dir', type=str, default=DEFAULT_SAVE_DIR,
                          help='Save Directorium')

        args.add_argument('-model', '--model_dir', type=str, default=PRETRAINED_MODEL_DIR,
                          help='Model Directorium')

        parsed_arg = args.parse_args()

        crawler = process(media_dir=parsed_arg.media_dir,
                          save_dir=parsed_arg.save_dir,
                          model_dir = parsed_arg.model_dir
                          )
