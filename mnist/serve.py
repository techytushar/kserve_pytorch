import base64
import io
import json
from typing import Dict, List

import kserve
import torch
from PIL import Image
from transforms import image_transforms


class MNISTModel(kserve.Model):
    def __init__(self, name: str, model_uri: str):
        super().__init__(name)
        self.name = name
        self.model_uri = model_uri
        self.ready = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_uri)

    def load_model(self, model_uri: str):
        model = torch.load(model_uri, map_location=self.device)
        model.eval()
        self.ready = True
        return model

    async def preprocess(
        self, inputs: bytes, headers: Dict[str, str] | None = None
    ) -> torch.Tensor:
        payload = json.loads(inputs)
        image = payload["instances"][0]["image_bytes"]
        image = Image.open(io.BytesIO(base64.b64decode(image)))
        return image_transforms(image).unsqueeze(0).repeat(1, 3, 1, 1)

    async def predict(
        self, input_tensor: torch.Tensor, headers: Dict[str, str] | None = None
    ) -> List:
        with torch.no_grad():
            output = self.model(input_tensor)
            return output.argmax(dim=1).tolist()

    async def postprocess(
        self, prediction: List, headers: Dict[str, str] | None = None
    ) -> Dict:
        return {"predictions": prediction}


if __name__ == "__main__":
    model = MNISTModel("mnist-model", "models/mnist_model.pth")
    kserve.ModelServer().start([model])
