import pdb
from collections import OrderedDict
import os
import pickle

import torch
import cv2
import torchvision
import numpy as np



class EmbeddingComputer:
    def __init__(self, dataset):
        self.model = None
        self.dataset = dataset
        self.crop_size = (128, 384)
        os.makedirs("./cache/embeddings/", exist_ok=True)
        self.cache_path = "./cache/embeddings/{}_embedding.pkl"
        self.cache = {}
        self.cache_name = ""

    def load_cache(self, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.cache = pickle.load(fp)

    def compute_embedding(self, img, bbox, tag, is_numpy=True):
        if self.cache_name != tag.split(":")[0]:
            self.load_cache(tag.split(":")[0])

        if tag in self.cache:
            embs = self.cache[tag]
            if embs.shape[0] != bbox.shape[0]:
                raise RuntimeError(
                    "ERROR: The number of cached embeddings don't match the "
                    "number of detections.\nWas the detector model changed? Delete cache if so."
                )
            return embs

        if self.model is None:
            self.initialize_model()

        # Make sure bbox is within image frame
        if is_numpy:
            h, w = img.shape[:2]
        else:
            h, w = img.shape[2:]
        results = np.round(bbox).astype(np.int32)
        results[:, 0] = results[:, 0].clip(0, w)
        results[:, 1] = results[:, 1].clip(0, h)
        results[:, 2] = results[:, 2].clip(0, w)
        results[:, 3] = results[:, 3].clip(0, h)

        # Generate all the crops
        crops = []
        for p in results:
            if is_numpy:
                crop = img[p[1] : p[3], p[0] : p[2]]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
                crop = torch.as_tensor(crop.astype("float32").transpose(2, 0, 1))
                crop = crop.unsqueeze(0)
            else:
                crop = img[:, :, p[1] : p[3], p[0] : p[2]]
                crop = torchvision.transforms.functional.resize(crop, self.crop_size)

            crops.append(crop)

        crops = torch.cat(crops, dim=0)

        # Create embeddings and l2 normalize them
        with torch.no_grad():
            crops = crops.cuda()
            crops = crops.half()
            embs = self.model(crops)
        embs = torch.nn.functional.normalize(embs)
        embs = embs.cpu().numpy()

        self.cache[tag] = embs
        return embs

    def initialize_model(self):
        """
        model = torchreid.models.build_model(name="osnet_ain_x1_0", num_classes=2510, loss="softmax", pretrained=False)
        sd = torch.load("external/weights/osnet_ain_ms_d_c.pth.tar")["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in sd.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()
        model.cuda()
        """
        if self.dataset == "mot17":
            path = "external/weights/mot17_sbs_S50.pth"
        elif self.dataset == "mot20":
            path = "external/weights/mot20_sbs_S50.pth"
        elif self.dataset == "dance":
            path = None
        else:
            raise RuntimeError("Need the path for a new ReID model.")

        model = FastReID(path)
        model.eval()
        model.cuda()
        model.half()
        self.model = model

    def dump_cache(self):
        if self.cache_name:
            with open(self.cache_path.format(self.cache_name), "wb") as fp:
                pickle.dump(self.cache, fp)
