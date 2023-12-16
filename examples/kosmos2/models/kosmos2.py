from typing import Any, Dict

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class Kosmos2:
    def __init__(self, model_name: str = "microsoft/kosmos-2-patch14-224") -> None:
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)

        self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def image_to_text(self, prompt: str, image: Image.Image, cleanup_and_extract: bool = True) -> Dict[str, Any]:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # By default, the generated  text is cleanup and the entities are extracted.
        processed_text, entities = self.processor.post_process_generation(
            generated_text, cleanup_and_extract=cleanup_and_extract
        )

        return {"processed_text": processed_text, "entities": entities}
