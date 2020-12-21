"""Prepare imageset for training and testing models."""

from pathlib import Path

from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte


class ImgPreprocessor:
    """Preprocessor for training and testing image set.

    Raw image set converter into structured data suitable for training and
    testing classification models. Can perform basic preparations and store
    images in directories according to their labels.
    """

    def __init__(self, img_size: int, root_dir: Path):
        """Initialize preprocessor with image set path.

        :param img_size: Image width and height (assumes square image ratio).
        :param root_dir: Image set root directory path.
        """
        self._img_size = (img_size, img_size)
        self._root_dir = root_dir
        self._sub_dirs = ['false', 'true']

    def __repr__(self):
        """Get information about image set found by this preprocessor."""
        img_num = len(sorted(Path(self._root_dir).glob('*/**/*.jpg')))
        return f'Found {img_num} images in the image set root directory.'

    def prepare_imgs(self, input_glob: str, output_dir: Path, class_gap: int):
        """Prepare image set for machine learning.

        :param input_glob: Glob expr to find images in the root directory.
        :param output_dir: Path to directory where output should be stored.
        :param class_gap: This assumes that in the root dir 'true' and 'false'
            labeled images are stored in alternating blocks of 'class_gap'
            length.
        """
        self._prepare_sub_dirs(output_dir)
        class_flag = False
        for i, img_path in enumerate(sorted(self._root_dir.glob(input_glob))):
            print(f'Iteration {i} of image set preparation.')
            img_name = img_path.name
            img = io.imread(img_path, pilmode='RGB')
            img = img_as_ubyte(resize(img, self._img_size, anti_aliasing=True))
            output_path = output_dir/self._sub_dirs[class_flag]/img_name
            io.imsave(output_path, img)

            if i % class_gap == class_gap - 1:
                class_flag = not class_flag

        return self

    def _prepare_sub_dirs(self, output_dir: Path):
        sub_dirs = [output_dir/sub for sub in self._sub_dirs]
        for dir_ in sub_dirs:
            dir_.mkdir(parents=True, exist_ok=True)
        return self


def main():
    """Prepare all images for training and testing a classifier."""
    img_prep = ImgPreprocessor(640, Path('data/raw'))
    print(img_prep)
    img_prep.prepare_imgs('*train*/**/*.jpg', Path('data/train'), 10)
    img_prep.prepare_imgs('*test*/**/*.jpg', Path('data/test'), 25)


if __name__ == '__main__':
    main()
