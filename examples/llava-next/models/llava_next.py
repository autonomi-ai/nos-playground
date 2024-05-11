from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class LlavaNextBase:
    """
    Base class for Llava-Next models, providing common methods for preprocessing.
    """
    def __init__(self):
        """
        Initializes the processor and model with pretrained weights.
        """
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def preprocess(self, image: Image.Image, prompt: str):
        """
        Preprocesses the image and prompt for the model.

        Parameters:
        image (PIL.Image.Image): The image to process.
        prompt (str): The prompt to accompany the image.

        Returns:
        A dictionary containing the processed image and prompt tensors.
        """
        return self.processor(text=prompt, images=image, return_tensors="pt", padding=True)

class LlavaNextImage(LlavaNextBase):
    """
    Llava-Next model for generating text from an image and a prompt.
    """
    def generate(self, image: Image.Image, prompt: str) -> str:
        """
        Generates text based on the input image and prompt.

        Parameters:
        image (PIL.Image.Image): The input image.
        prompt (str): The input prompt.

        Returns:
        str: The generated text.
        """
        # Validate inputs
        if not isinstance(image, Image.Image):
            raise ValueError("The image must be a PIL.Image.Image instance")
        if not isinstance(prompt, str):
            raise ValueError("The prompt must be a string")

        # Preprocess the image and prompt
        inputs = self.preprocess(image, prompt)

        # Generate text from image and prompt
        outputs = self.model(**inputs)

        # Placeholder for actual text generation logic
        generated_text = "Generated text based on the image and prompt"
        return generated_text

class LlavaNextVideo(LlavaNextBase):
    """
    Llava-Next model for generating text from a video and a prompt.
    """
    def generate(self, image: Image.Image, prompt: str) -> str:
        """
        Generates text based on the input video and prompt.

        Parameters:
        image (PIL.Image.Image): The input image representing the video.
        prompt (str): The input prompt.

        Returns:
        str: The generated text.
        """
        # Validate inputs
        if not isinstance(image, Image.Image):
            raise ValueError("The image must be a PIL.Image.Image instance")
        if not isinstance(prompt, str):
            raise ValueError("The prompt must be a string")

        # Preprocess the image and prompt
        inputs = self.preprocess(image, prompt)

        # Generate text from image and prompt
        outputs = self.model(**inputs)

        # Placeholder for actual text generation logic
        generated_text = "Generated text based on the video and prompt"
        return generated_text
