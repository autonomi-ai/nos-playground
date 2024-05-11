import pytest
import os
from PIL import Image
from models.llava_next import LlavaNextImage, LlavaNextVideo

@pytest.fixture
def sample_image():
    # Assuming there is a sample image in the directory for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "sample_image.png")
    return Image.open(image_path)

@pytest.fixture
def sample_prompt():
    return "A sample prompt for testing."

def test_llava_next_image_generate(sample_image, sample_prompt):
    model = LlavaNextImage()
    result = model.generate(sample_image, sample_prompt)
    assert isinstance(result, str)
    assert result == "Generated text based on the image and prompt"  # Expected result as per current implementation

def test_llava_next_video_generate(sample_image, sample_prompt):
    model = LlavaNextVideo()
    result = model.generate(sample_image, sample_prompt)
    assert isinstance(result, str)
    assert result == "Generated text based on the video and prompt"  # Expected result as per current implementation
