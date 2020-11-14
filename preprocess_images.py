import os
import glob
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte


class ImagesPreprocess:
    def __init__(self, size: int, parent_dir: str):
        self.__size = (size, size)
        self.__parent_dir = parent_dir
        self.__train_dir = f"{parent_dir}/train"
        self.__sub_dirs = ["false", "true"]

    @staticmethod
    def __create_dir(path: str):
        try:
            os.mkdir(path)
        except FileExistsError:
            print(f"{path} -- directory already exists")

    def create_model_dirs(self):
        sub_dirs = [f"{self.__train_dir}/{sub}" for sub in self.__sub_dirs]
        for dir_ in [self.__train_dir, *sub_dirs]:
            ImagesPreprocess.__create_dir(dir_)
        return self

    def preprocess_images(self):
        path_pattern = f"{self.__parent_dir}/[s]z*/**/*.jpg"
        flag = False
        for i, img_path in enumerate(glob.glob(path_pattern, recursive=True)):
            img_name = os.path.basename(img_path)
            img = io.imread(img_path, pilmode="RGB")
            img = img_as_ubyte(resize(img, self.__size, anti_aliasing=True))
            out_path = f"{self.__train_dir}/{self.__sub_dirs[flag]}/{img_name}"
            io.imsave(out_path, img)
            if i % 10 == 9:
                flag = not flag
        return self


def main():
    images_path = "../images"
    ImagesPreprocess(640, images_path)\
        .create_model_dirs()\
        .preprocess_images()


if __name__ == "__main__":
    main()
