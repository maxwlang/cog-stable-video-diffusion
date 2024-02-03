# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import json
import os
import urllib
import urllib.request

import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from torchvision import transforms

if "TOKEN" not in os.environ:
    TOKEN = ''
    print('Missing TOKEN environment variable.')
else:
    TOKEN = os.environ['TOKEN']


class Predictor(BasePredictor):
    def load_model(self) -> torch.nn.Module:
        model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
        model.to(self.device)
        model.eval()

        return model

    def load_labels(self):
        with urllib.request.urlopen("https://github.com/RF5/danbooru-pretrained/raw/master/config/class_names_6000.json") as url:
            return json.loads(url.read())

    def setup(self):
        print("Loading pipeline...")
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.labels = self.load_labels()
        self.preprocess = transforms.Compose([
            transforms.Resize(360),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7137, 0.6628, 0.6519],
                                 std=[0.2970, 0.3017, 0.2979]),
        ])

    @torch.inference_mode()
    def predict(
        self,
        inputImage: Path = Input(
            description="The image we will extract tags from"),
        scoreThreshold: float = Input(
            description="The minimum threshold to accept as valid", ge=0, le=1, default=0.4
        ),
    ) -> dict:
        """Run a single prediction on the model"""
        image = Image.open(inputImage).convert("RGB")
        inputTensor = self.preprocess(image)
        # create a mini-batch as expected by the model
        inputBatch = inputTensor.unsqueeze(0)

        if torch.cuda.is_available():
            inputBatch = inputBatch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(inputBatch)

        # Tensor of shape 6000, with confidence scores over Danbooru's top 6000 tags
        probs = torch.sigmoid(output[0])
        res = dict()
        for prob, label in zip(probs.tolist(), self.labels):
            if prob < scoreThreshold:
                continue
            res[label] = prob
        return res
